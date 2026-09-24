"""Opt-in joint diagnostics and complete adaptation resume for COLMAP runs."""
import json
import random
import numpy as np
import torch
from scipy.spatial import cKDTree
from training.trainer import Trainer
from training.checkpoints import checkpoint_payload, save_checkpoint, load_checkpoint

ACCUMULATORS=('contribution','visibility','path_length','weighted_path_length','max_error','gradient')

def map_birth_steps(birth,plan,iteration):
    return torch.cat((birth[plan.keep_indices],birth.new_full((plan.clone_count+2*plan.split_count,),iteration)))


class ColmapJointTrainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.birth_steps=torch.zeros(self.model.gaussians.point_count,dtype=torch.long,device=self.device)
        self.joint_iteration=0
        if self.resume_path is not None:
            payload=load_checkpoint(self.resume_path)
            extra=payload.get('colmap_joint')
            if extra is None: raise ValueError('Missing COLMAP joint resume state')
            self.birth_steps=extra['birth_steps'].to(self.device)
            if len(self.birth_steps)!=self.model.gaussians.point_count: raise ValueError('Invalid lineage')
            perm=self.topology.last_permutation
            self.voronoi_adaptation.reorder_statistics(perm)
            for name,value in extra['medium_accumulators'].items():
                setattr(self.voronoi_adaptation,name,value.to(self.device)[perm].contiguous())
            self.sampler._camera_stack=list(extra['camera_stack'])
            random.setstate(extra['python_rng']);np.random.set_state(extra['numpy_rng'])
        points,_=self.dataset.initial_point_cloud()
        self.colmap_tree=cKDTree(points)
        controller=self.gaussian_adaptation
        original=controller.apply_plan
        def apply(plan):
            birth=self.birth_steps if plan.is_noop else map_birth_steps(self.birth_steps,plan,self.joint_iteration)
            result=original(plan)
            self.birth_steps=birth
            return result
        controller.apply_plan=apply

    def _train_step(self,iteration):
        self.joint_iteration=iteration
        return super()._train_step(iteration)

    def _checkpoint(self,path,iteration):
        payload=checkpoint_payload(self.model,self.gaussian_optimizer,self.medium_optimizer,
            self.config,iteration,self.training_seconds,self.best_psnr,self.topology.revision,
            self.gaussian_adaptation.state_dict(),self.voronoi_adaptation.state_dict(),
            self.medium_frozen_until,self.disentanglement.state_dict(),self.prefix_training_seconds)
        payload['colmap_joint']=dict(birth_steps=self.birth_steps.detach().cpu(),
            medium_accumulators={n:getattr(self.voronoi_adaptation,n).detach().cpu() for n in ACCUMULATORS},
            camera_stack=list(self.sampler._camera_stack),python_rng=random.getstate(),numpy_rng=np.random.get_state())
        save_checkpoint(path,payload)

    def _evaluate_and_checkpoint(self,iteration):
        super()._evaluate_and_checkpoint(iteration)
        with torch.no_grad():
            g=self.model.gaussians
            distance=self.colmap_tree.query(g.means.detach().cpu().numpy(),workers=4)[0]
            born=self.birth_steps.cpu().numpy()>0
            opacity=g.opacities.flatten().cpu().numpy()
            report=dict(iteration=iteration)
            for name,mask in [('initial',~born),('spawned',born)]:
                report[name]=dict(count=int(mask.sum()),
                    distance_quantiles=np.quantile(distance[mask],[.5,.9,.99]).tolist() if mask.any() else [],
                    opacity_quantiles=np.quantile(opacity[mask],[.1,.5,.9]).tolist() if mask.any() else [])
            folder=self.output_dir/'colmap_diagnostics';folder.mkdir(exist_ok=True)
            (folder/f'iteration_{iteration:06d}.json').write_text(json.dumps(report,indent=2))
