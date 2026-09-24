"""Separate trainer selection: imported ONLY by the new experiment entrypoint."""
from dataclasses import replace
import json
from pathlib import Path
from training.colmap_joint_trainer import ColmapJointTrainer
from training.disentanglement import undirected_edges
from training.end_to_end import atomic_json
from experiments.rgb_smooth.loss import SmoothConfig, smooth_loss


class SmoothJointTrainer(ColmapJointTrainer):
    def __init__(self,config,*args,smoothing=None,**kwargs):
        if smoothing is None:
            manifest=json.loads((Path(config.output_dir).parent/'pipeline_config.json').read_text())
            smoothing=manifest['rgb_smoothing']
            if manifest['warmup_steps']!=0:raise ValueError('Smoothing sweep has no warmup')
        self.smoothing=SmoothConfig(**smoothing)
        self.smooth_metrics={}
        self._smooth_edge_key=None
        self._smooth_edges=None
        super().__init__(config,*args,**kwargs)
        if not self.model.medium.is_rgb:raise ValueError('Smoothing trainer requires RGB medium')
        if self.resume_path is not None:
            saved=json.loads(Path(self.resume_path).with_suffix('.smooth.json').read_text())
            if saved!=self.smoothing.as_dict():raise ValueError('Resume smoothing settings mismatch')
        original=self.disentanglement.compute
        def compute(iteration,model,topology,*args,**kwargs):
            result=original(iteration,model,topology,*args,**kwargs)
            key=(topology.revision,topology.adjacency.data_ptr(),topology.offsets.data_ptr(),model.medium.point_count)
            if key!=self._smooth_edge_key:
                self._smooth_edges=undirected_edges(topology.adjacency,topology.offsets,model.medium.point_count)
                self._smooth_edge_key=key
            extra,parts=smooth_loss(model.medium.optical_coefficients,model.medium.colors,
                                   *self._smooth_edges,self.smoothing,iteration)
            self.smooth_metrics={f'rgb_smooth/{k}':float(v.detach()) if hasattr(v,'detach') else v for k,v in parts.items()}
            self.smooth_metrics['rgb_smooth/weighted']=float(extra.detach())
            return replace(result,total=result.total+extra)
        self.disentanglement.compute=compute

    def _training_row(self,*args):
        return {**super()._training_row(*args),**self.smooth_metrics}

    def _checkpoint(self,path,iteration):
        super()._checkpoint(path,iteration)
        atomic_json(Path(path).with_suffix('.smooth.json'),self.smoothing.as_dict())
