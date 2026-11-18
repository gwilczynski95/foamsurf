"""Compare a frozen Lego checkpoint to Blender insertion GT; no training or LPIPS.

The checkpoint learns PNG-space radiance, so the inserted emission is encoded in
sRGB before passing it to that renderer. This is not linear-light equivalence.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter
import torch


def run(args):
    import foamsurf._C as native
    import render
    from evaluation.metrics import ssim
    from .tracer import SceneTracer, render_camera
    from .shading import Sphere, TraceSettings
    if args.scipy_topology:
        import foamsurf_topology
        from scipy.spatial import Delaunay
        class ExactSitesTopology:
            def __init__(self,sites):
                print('TOPOLOGY scipy float64, unchanged sites',sites.shape,flush=True)
                triangulation=Delaunay(sites.detach().cpu().double().numpy())
                if len(triangulation.coplanar):
                    raise RuntimeError('Qhull omitted sites; refusing incomplete topology')
                offsets,adjacency=triangulation.vertex_neighbor_vertices
                self.adj=torch.tensor(adjacency.astype(np.int64),device=sites.device).to(torch.uint32)
                self.off=torch.tensor(offsets.astype(np.int64),device=sites.device).to(torch.uint32)
                self.perm=torch.arange(len(sites),device=sites.device)
            def permutation(self):return self.perm
            def point_adjacency(self):return self.adj
            def point_adjacency_offsets(self):return self.off
        foamsurf_topology.Triangulation=ExactSitesTopology

    if not hasattr(native, 'rgb_forward_limited'):
        raise RuntimeError(f'Wrong native extension: {native.__file__}')
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f'Refusing to overwrite {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    joint = args.run/'joint' if (args.run/'joint').is_dir() else args.run
    checkpoint = joint/'checkpoints'/'last.pt'
    print('LOAD', checkpoint, 'native', native.__file__, flush=True)
    trainer = render.load_run(args.run.resolve(), scene=args.scene)
    config = trainer.config
    if trainer.camera_isp is not None or trainer.evaluator.water_light is not None:
        raise RuntimeError('This controlled experiment expects no ISP or water-light correction')

    class BackgroundSceneTracer(SceneTracer):
        """Restore the terminal background omitted by the branch's black-background tracer.

        A finite segment ending at the inserted object has no background term.
        A segment extending to the far plane has T * checkpoint_background.
        Checked below against the normal native renderer for every camera.
        """
        @torch.no_grad()
        def segment(self, rays, far, start_points=None):
            direct, scattered, transmission = super().segment(rays, far, start_points)
            bg = torch.tensor(config.background, device=rays.device, dtype=rays.dtype)
            terminal = (far >= self.renderer.far)[:, None]
            return direct + terminal * transmission * bg, scattered, transmission

    info = json.loads((args.gt/'insert.json').read_text())
    if (info['view_transform'], info['look'], info['exposure'], info['gamma']) != ('Standard', 'None', 0.0, 1.0):
        raise RuntimeError('Unsupported Blender display transform')
    linear = np.array(info['color']) * info['strength']
    encoded = np.where(linear <= 0.0031308, 12.92*linear, 1.055*linear**(1/2.4)-0.055)
    sphere = Sphere(tuple(info['location']), info['radius'], 'emissive', tuple(encoded),
                    shadow=False, power=math.pi*info['radius']**2)
    settings = TraceSettings(max_depth=1, shadows=False, lamps=False, water_light_max_up=None,
                             antialias=8, min_weight=0.0)
    tracer = BackgroundSceneTracer(trainer)
    cameras = [c for c in trainer.dataset.test if c.name in info['views']]
    if args.benchmark_views:
        wanted = set(args.benchmark_views.split(','))
        cameras = [c for c in cameras if c.name in wanted]
        if {c.name for c in cameras} != wanted:
            raise RuntimeError('Requested cameras absent from checkpoint dataset')
    elif {c.name for c in cameras} != set(info['views']):
        raise RuntimeError('GT/checkpoint camera names differ')
    source_cameras = json.loads((args.gt/'transforms_test.json').read_text())
    source_frames = {Path(f['file_path']).name:f for f in source_cameras['frames']}

    def rgba_target(path):
        a = torch.as_tensor(np.array(Image.open(path).convert('RGBA'),np.float32)/255, device=trainer.device)
        bg = torch.tensor(config.background,device=trainer.device)
        return a[...,:3]*a[...,3:] + bg*(1-a[...,3:])

    def score(pred,target,region=None):
        delta=(pred-target) if region is None else (pred-target)[region]
        mse=float((delta**2).mean())
        return dict(psnr_db=-10*math.log10(max(mse,1e-12)),mae=float(delta.abs().mean()),mse=mse)

    def save(folder,name,im):
        dest=args.out/folder
        dest.mkdir(exist_ok=True)
        Image.fromarray(render.to_png('full',im)).save(dest/f'{name}.png')
        if folder in {'prediction','baseline'}:
            np.save(dest/f'{name}.npy',im.detach().float().cpu().numpy())

    results={}
    provenance=dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_iteration=trainer.start_iteration-1, native_extension=str(native.__file__),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        background=config.background, color_convention='checkpoint PNG/sRGB-space; analytic sRGB encoding of Blender emission, no fit',
        emission_linear=linear.tolist(),emission_encoded=encoded.tolist(),
        sphere_center=info['location'],sphere_radius=info['radius'], antialias=8,
        roi='full Blender silhouette alpha > 0.5, including occluded pixels',
        baseline='normal renderer, frozen last checkpoint, original images without sphere',
        topology='scipy float64, unchanged sites' if args.scipy_topology else 'native',
        lpips_run=False)
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('LOADED',json.dumps(provenance),flush=True)
    with torch.no_grad():
        for camera in cameras:
            start=time.monotonic()
            matrix=np.asarray(source_frames[camera.name]['transform_matrix'])
            expected=[matrix[:3,3],matrix[:3,0],-matrix[:3,1],-matrix[:3,2]]
            actual=[camera.origin,camera.right,camera.down,camera.forward]
            camera_error=max(float(np.abs(x.detach().cpu().numpy()-y).max()) for x,y in zip(actual,expected))
            if camera_error>1e-5 or abs(camera.fov_x-source_cameras['camera_angle_x'])>1e-6:
                raise RuntimeError(f'Camera mismatch {camera.name}: {camera_error}')
            baseline=render.render_view(trainer,camera,['full'])['full']
            _,empty=render_camera(tracer,camera,[],settings)
            equivalence=float((baseline-empty).abs().max())
            if equivalence>1e-5:
                raise RuntimeError(f'Empty tracer mismatch {camera.name}: {equivalence}')
            _,prediction=render_camera(tracer,camera,[sphere],settings)
            target=rgba_target(args.gt/'gt_rgba'/f'{camera.name}.png')
            reference=camera.ground_truth(tuple(config.background))
            mask_image=Image.open(args.gt/'masks'/f'{camera.name}.png').getchannel('A')
            mask=torch.tensor(np.array(mask_image)>127,device=trainer.device)
            outside=torch.tensor(np.array(mask_image.filter(ImageFilter.MaxFilter(7)))==0,device=trainer.device)
            record=dict(camera_error=camera_error,empty_tracer_max_error=equivalence,
                        baseline_full=score(baseline,reference),baseline_roi=score(baseline,reference,mask),
                        insertion_full=score(prediction,target),insertion_roi=score(prediction,target,mask),
                        outside_vs_baseline=score(prediction,baseline,outside),
                        outside_max_abs=float((prediction-baseline)[outside].abs().max()),roi_pixels=int(mask.sum()))
            record['baseline_full']['ssim']=float(ssim(baseline,reference))
            record['insertion_full']['ssim']=float(ssim(prediction,target))
            for folder,im in [('baseline',baseline),('prediction',prediction),('gt',target),('original',reference)]:
                save(folder,camera.name,im)
            results[camera.name]=record
            (args.out/'per_view.json').write_text(json.dumps(results,indent=2))
            print(camera.name,'baseline',round(record['baseline_full']['psnr_db'],3),
                  'insertion ROI',round(record['insertion_roi']['psnr_db'],3),
                  'seconds',round(time.monotonic()-start,2),flush=True)
    summary={}
    for region in ['baseline_full','baseline_roi','insertion_full','insertion_roi']:
        summary[region]={key:float(np.mean([r[region][key] for r in results.values()])) for key in results[next(iter(results))][region]}
    summary['num_views']=len(results)
    summary['max_empty_tracer_error']=max(r['empty_tracer_max_error'] for r in results.values())
    summary['max_outside_change']=max(r['outside_max_abs'] for r in results.values())
    roi = [r['insertion_roi'] for r in results.values()]
    pixels = sum(r['roi_pixels'] for r in results.values())
    pooled_mse = sum(r['insertion_roi']['mse'] * r['roi_pixels'] for r in results.values()) / pixels
    metrics = dict(base_psnr=summary['baseline_full']['psnr_db'],
                   roi_psnr=summary['insertion_roi']['psnr_db'],
                   roi_pooled_psnr=-10 * math.log10(max(pooled_mse, 1e-12)),
                   roi_mae=summary['insertion_roi']['mae'])
    (args.out/'metrics.json').write_text(json.dumps(dict(mean=metrics, views=results, checkpoint='last',
                                                       evaluation='insertion'), indent=2))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.out/'COMPLETE').write_text(f'{len(results)} views\n')
    print('COMPARISON_DONE',json.dumps(summary),flush=True)


