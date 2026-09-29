"""Only the new entrypoint selects this trainer; historical runs are preserved."""
from dataclasses import replace
import json
from pathlib import Path

from experiments.rgb_smooth.trainer import SmoothJointTrainer
from training.end_to_end import atomic_json
from .loss import SurfaceConfig
from .renderer import LocalOverlap
from .coverage_artifacts import install_coverage


class SurfaceTrainer(SmoothJointTrainer):
    def __init__(self, config, *args, surface=None, **kwargs):
        if surface is None:
            manifest = json.loads((Path(config.output_dir).parent/'pipeline_config.json').read_text())
            surface = manifest['rgb_surface']
            if manifest['warmup_steps'] != 0:
                raise ValueError('Surface sweep requires zero warmup')
        self.surface = SurfaceConfig(**surface)
        self.surface_metrics = {}
        self._captured_surface = None
        self._capture_surface = False
        self._overlap = LocalOverlap()
        super().__init__(config, *args, **kwargs)
        if self.resume_path is not None:
            if json.loads(Path(self.resume_path).with_suffix('.surface.json').read_text()) != self.surface.as_dict():
                raise ValueError('Resume overlap settings mismatch')
        native_render = self.renderer.render

        def capture(rays, *a, **kw):
            result = native_render(rays, *a, **kw)
            if self._capture_surface and self._captured_surface is None and kw.get('gaussians') is not None and kw.get('voronoi') is not None:
                self._captured_surface = (rays, kw['gaussians'])
            return result

        self.renderer.render = capture
        compute_base = self.disentanglement.compute

        def compute(iteration, model, topology, *a, **kw):
            result = compute_base(iteration, model, topology, *a, **kw)
            if self._captured_surface is None:
                raise RuntimeError('No hybrid training render captured')
            extra, logs = self._overlap.compute(native_render, *self._captured_surface,
                model.medium, topology, self.surface, iteration, self.scene_extent,
                self.config.renderer.far, self.config.renderer.near, self.config.seed)
            self.surface_metrics = {f'surface_overlap/{k}': v for k, v in logs.items()}
            return replace(result, total=result.total+extra)

        self.disentanglement.compute = compute
        install_coverage(self.evaluator)

    def _train_step(self, iteration):
        self._captured_surface = None
        self._capture_surface = True
        try:
            return super()._train_step(iteration)
        finally:
            self._capture_surface = False
            self._captured_surface = None

    def _training_row(self, *args):
        return {**super()._training_row(*args), **self.surface_metrics}

    def _checkpoint(self, path, iteration):
        super()._checkpoint(path, iteration)
        atomic_json(Path(path).with_suffix('.surface.json'), self.surface.as_dict())
