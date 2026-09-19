"""Partition a native warmup and initialize the standard hybrid representation."""
from pathlib import Path
import numpy as np
import torch

from evaluation.surface_partition import classify, variants, checkpoint_order
from training.end_to_end import atomic_json
from training.radfoam_handoff import _surface_orientations, _cells_touching_surface, _inverse_softplus


def local_tv(sigma, adjacency, offsets, outer):
    """Same inner-inner mean absolute contrast as the validated analyzer."""
    sigma = sigma.detach().cpu().flatten()
    outer = outer.detach().cpu().bool()
    adjacency, offsets = adjacency.cpu().long(), offsets.cpu().long()
    degree = offsets[1:]-offsets[:-1]
    owners = torch.repeat_interleave(torch.arange(len(sigma)), degree)
    keep = ~outer[owners] & ~outer[adjacency]
    owners, neighbors = owners[keep], adjacency[keep]
    total = torch.zeros_like(sigma).index_add_(0, owners, (sigma[owners]-sigma[neighbors]).abs())
    count = torch.zeros_like(sigma).index_add_(0, owners, torch.ones(len(owners)))
    return total / count.clamp_min(1)


def handoff_plan(surface, adjacency, offsets, checkpoint_index, gaussian_count, seed):
    """Export masks and sampling indices in source checkpoint order."""
    surface = torch.as_tensor(surface, dtype=torch.bool, device='cpu')
    if not surface.any():
        raise ValueError('No confident SURFACE found; refusing Gaussian initialization')
    if int((~surface).sum()) < 4:
        raise ValueError('Fewer than four retained Voronoi sites')
    affected = _cells_touching_surface(surface, adjacency, offsets)
    source_surface = torch.from_numpy(checkpoint_order(surface.numpy(), checkpoint_index))
    source_affected = torch.from_numpy(checkpoint_order(affected.numpy(), checkpoint_index))
    indices = source_surface.nonzero().flatten()
    count = min(gaussian_count, len(indices))
    if count < 1:
        raise ValueError('Gaussian count must be positive')
    rng = torch.Generator().manual_seed(seed)
    selected = indices[torch.randperm(len(indices), generator=rng)[:count]]
    return dict(surface_mask=source_surface, affected_mask=source_affected,
                gaussian_source_indices=selected, order='source_checkpoint')


@torch.no_grad()
def partition_warmup(root, manifest):
    from training.config import ExperimentConfig
    from training.trainer import Trainer
    from training.checkpoints import save_checkpoint
    from scripts.analyze_voronoi_warmup import _rays
    from scripts.compare_surface_partitions import plot_partition
    config = ExperimentConfig.load(root/'warmup.yaml')
    config.output_dir = str(root/'partition/context')
    config.wandb.mode = 'disabled'
    settings = manifest['partition']
    checkpoint = root/'warmup/checkpoints/last.pt'
    trainer = Trainer(config, resume=checkpoint)
    output = root/'partition'
    output.mkdir(exist_ok=True)
    try:
        trainer.renderer.transmittance_threshold = config.renderer.validation_transmittance_threshold
        medium = trainer.model.medium
        contribution = torch.zeros(medium.point_count, dtype=torch.float32)
        scene = medium.render_scene()
        chunk = min(settings['batch_size'], trainer.renderer.max_batch_size)
        # Full TRAIN camera support; neither test views nor layer GT affect labels.
        for camera in trainer.dataset.train:
            rays = _rays(camera).reshape(-1,6)
            for start in range(0,len(rays),chunk):
                rendered = trainer.renderer.render(rays[start:start+chunk], voronoi=scene,
                    background=tuple(config.background), collect_diagnostics=True)
                if bool((rendered.diagnostics.traversal_truncated > 0).any()):
                    raise RuntimeError('Truncated classification rays; increase max_intersections')
                contribution += rendered.diagnostics.medium[:,0].detach().cpu()
                del rendered
        sigma = medium.densities.flatten().cpu()
        outer = medium.guard_mask.cpu()
        tv = local_tv(sigma, trainer.topology.adjacency, trainer.topology.offsets, outer)
        features = dict(sigma=sigma.numpy(), local_tv=tv.numpy(), contribution=contribution.numpy(),
                        outer_mask=outer.numpy())
        variant = next(s for s in variants() if s['name']==settings['method'])
        result = classify(**dict(sigma=features['sigma'],tv=features['local_tv'],
                            contribution=features['contribution'],outer=features['outer_mask']), **variant)
        plot_partition(features, result, output)
        atomic_json(output/'classifier.json', result.info)
        if result.info['status'] != 'candidate':
            raise RuntimeError(f'Uncertain surface partition: {result.info["reason"]}')
        plan = handoff_plan(result.surface, trainer.topology.adjacency,
                            trainer.topology.offsets, trainer.topology.last_permutation.cpu().numpy(),
                            manifest['handoff']['gaussian_count'], config.seed)
        plan['source_checkpoint'] = str(checkpoint)
        plan['source_sites'] = torch.from_numpy(checkpoint_order(
            medium.sites.detach().cpu().numpy(), trainer.topology.last_permutation.cpu().numpy()))
        plan['classifier'] = result.info
        artifact = output/'handoff.pt'
        save_checkpoint(artifact, plan)
        _render_partition(trainer, result.surface, output, settings)
        return artifact
    finally:
        trainer.logger.close()


@torch.no_grad()
def _render_partition(trainer, surface, output, settings):
    from scripts.analyze_voronoi_warmup import _rays, _masked_scene, _label_scene, _grid, _to_image
    labels = torch.where(torch.as_tensor(surface, device=trainer.device),1,2)
    for i,camera in enumerate(trainer.dataset.test[:settings['image_views']]):
        scenes = [('full warmup',trainer.model.medium.render_scene()),
                  ('SURFACE',_masked_scene(trainer.model,labels,1)),
                  ('NON-SURFACE',_masked_scene(trainer.model,labels,2)),
                  ('red SURFACE / blue NON-SURFACE',_label_scene(trainer.model,labels))]
        row = [('target',_to_image(camera.ground_truth(tuple(trainer.config.background))))]
        rays = _rays(camera)
        for name,scene in scenes:
            image = _chunk_image(trainer, rays, settings['batch_size'], voronoi=scene)
            row.append((name,_to_image(image)))
        _grid([row[:3],row[3:]]).save(output/f'view_{i:02d}_partition.png')


@torch.no_grad()
def _chunk_image(trainer, rays, batch_size, **scenes):
    shape = rays.shape[:-1]
    flat = rays.reshape(-1,6)
    pixels = []
    for start in range(0,len(flat),min(batch_size,trainer.renderer.max_batch_size)):
        chunk = flat[start:start+min(batch_size,trainer.renderer.max_batch_size)]
        image = trainer.renderer.render(chunk, background=tuple(trainer.config.background), **scenes).image
        pixels.append(image.detach().cpu().reshape(-1,3))
    return torch.cat(pixels).reshape(*shape,3)


@torch.no_grad()
def initialize_native_hybrid(root, config, options):
    from training.checkpoints import load_checkpoint
    from foamsurf_model.gaussians import SurfaceGaussianModel
    from foamsurf_model.medium import VoronoiMediumModel
    from foamsurf_model.scene import HybridSceneModel
    artifact = torch.load(root/'partition/handoff.pt',map_location='cpu',weights_only=False)
    checkpoint = root/'warmup/checkpoints/last.pt'
    if Path(artifact['source_checkpoint']).resolve()!=checkpoint.resolve() or artifact['order']!='source_checkpoint':
        raise ValueError('Invalid handoff checkpoint provenance')
    payload = load_checkpoint(checkpoint)
    if payload['config']['representation_mode']!='voronoi_only':
        raise ValueError('Expected a native Voronoi-only checkpoint')
    state = payload['model']
    sites = state['medium.sites']
    if not torch.allclose(sites,artifact['source_sites'],rtol=1e-5,atol=1e-5):
        raise ValueError('Handoff features do not match the source checkpoint geometry')
    surface = artifact['surface_mask']
    keep = ~surface
    selected = artifact['gaussian_source_indices']
    if not bool(surface[selected].all()) or len(selected)!=len(torch.unique(selected)):
        raise ValueError('Invalid Gaussian source indices')
    affected_kept = artifact['affected_mask'][keep]
    config.model.gaussian_initial_points = len(selected)
    config.model.medium_points = int(keep.sum())
    config.model.medium_guard_points = 0
    config.voronoi_adaptation.min_points = int(keep.sum())
    config.validate()
    gaussians = SurfaceGaussianModel(sites[selected].numpy(),
                torch.sigmoid(state['medium.raw_color'][selected]).numpy(),config.model,config.device)
    orientations = _surface_orientations(sites[surface].numpy(),sites[selected].numpy(),options['pca_neighbors'])
    gaussians.quaternions.copy_(torch.from_numpy(orientations).to(config.device))
    # Native warmup checkpoints already store beta=1 inverse of physical sigma.
    # Do not apply beta=10 a second time. Reset only cells touching removed sites.
    density = state['medium.raw_density'][keep].clone()
    density[affected_kept] = _inverse_softplus(options['reset_density'])
    medium = VoronoiMediumModel.from_checkpoint(
        dict(sites=sites[keep].clone(),raw_density=density,
             raw_color=state['medium.raw_color'][keep].clone()),
        dict(point_count=int(keep.sum()),guard_count=0),state['medium.bbox_min'],state['medium.bbox_max'],config.device)
    report = dict(source_count=len(sites),removed_surface_count=int(surface.sum()),
                  retained_voronoi_count=int(keep.sum()),reset_neighbor_count=int(affected_kept.sum()),
                  gaussian_count=len(selected),requested_gaussian_count=options['gaussian_count'],
                  reset_density=options['reset_density'],all_retained_cells_trainable=True,
                  classifier=artifact['classifier'],joint_start_iteration=0)
    return HybridSceneModel(gaussians,medium), report


@torch.no_grad()
def render_transition(trainer, output, settings):
    from scripts.analyze_voronoi_warmup import _rays, _to_image, _grid
    output.mkdir(exist_ok=True)
    trainer.topology.synchronize_renderer(trainer.renderer,trainer.model.medium)
    medium = trainer.model.medium.render_scene()
    for i,camera in enumerate(trainer.dataset.test[:settings['image_views']]):
        rays = _rays(camera)
        gaussian = trainer.model.gaussians.render_scene(camera.origin)
        row = [('target',_to_image(camera.ground_truth(tuple(trainer.config.background))))]
        for title,scenes in [('initialized hybrid',dict(gaussians=gaussian,voronoi=medium)),
                             ('Gaussians',dict(gaussians=gaussian)),('retained Voronoi',dict(voronoi=medium))]:
            row.append((title,_to_image(_chunk_image(trainer,rays,settings['batch_size'],**scenes))))
        _grid([row[:2],row[2:]]).save(output/f'view_{i:02d}_initialization.png')
