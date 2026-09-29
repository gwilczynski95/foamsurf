"""Instance-local loss and renderer integration; not imported by existing jobs."""
from dataclasses import replace
import json
from pathlib import Path
from experiments.rgb_smooth.trainer import SmoothJointTrainer
from training.end_to_end import atomic_json
from .losses import ConflictConfig
from .renderer import compute_regularizers


class ConflictTrainer(SmoothJointTrainer):
    def __init__(self,config,*args,conflict=None,**kwargs):
        if conflict is None:
            m=json.loads((Path(config.output_dir).parent/'pipeline_config.json').read_text())
            conflict=m['rgb_conflict']
            if m['warmup_steps']!=0:raise ValueError('No warmup supported')
        self.conflict=ConflictConfig(**conflict)
        self.conflict_metrics={}
        self._capture=False
        self._captured=None
        super().__init__(config,*args,**kwargs)
        if self.resume_path is not None:
            saved=json.loads(Path(self.resume_path).with_suffix('.conflict.json').read_text())
            if saved!=self.conflict.as_dict():raise ValueError('Resume conflict settings mismatch')
        original_render=self.renderer.render
        def capture(rays,*a,**kw):
            out=original_render(rays,*a,**kw)
            if self._capture and self._captured is None and kw.get('gaussians') is not None and kw.get('voronoi') is not None:
                self._captured=(rays,kw['gaussians'])
            return out
        self.renderer.render=capture
        original_compute=self.disentanglement.compute
        def compute(iteration,model,topology,*a,**kw):
            result=original_compute(iteration,model,topology,*a,**kw)
            if self._captured is None:raise RuntimeError('Missing hybrid training render for conflict losses')
            rays,scene=self._captured
            extra,logs=compute_regularizers(original_render,rays,scene,model.medium,self.conflict,
                iteration,self.scene_extent,self.config.renderer.far,self.config.seed)
            self.conflict_metrics={f'conflict/{k}':v for k,v in logs.items()}
            return replace(result,total=result.total+extra)
        self.disentanglement.compute=compute

    def _train_step(self,iteration):
        self._captured=None;self._capture=True
        try:return super()._train_step(iteration)
        finally:self._capture=False;self._captured=None

    def _training_row(self,*a):
        return {**super()._training_row(*a),**self.conflict_metrics}

    def _checkpoint(self,path,iteration):
        super()._checkpoint(path,iteration)
        atomic_json(Path(path).with_suffix('.conflict.json'),self.conflict.as_dict())
