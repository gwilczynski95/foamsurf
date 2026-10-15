"""Native warmup -> automatic partition -> fresh hybrid training in one fit().

Stages run in separate processes to release CUDA allocations between them.
Only run-stage implementation imports the GPU trainer. Configuration and
orchestration are usable on CPU; no external RadFoam checkpoint is involved.
"""
from copy import deepcopy
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml
from training.config import ExperimentConfig, drop_legacy_keys
from training.warmup_profile import WarmupProfile
from evaluation.surface_partition import variants

ROOT = Path(__file__).resolve().parents[1]


def update_config(obj, changes):
    """Strict recursive overrides, with the same field names as stage YAMLs."""
    names = {f.name for f in fields(obj)}
    for key, value in changes.items():
        if key not in names:
            raise ValueError(f'Unknown configuration field: {key}')
        if is_dataclass(getattr(obj, key)):
            if not isinstance(value, dict):
                raise ValueError(f'{key} must be a mapping')
            update_config(getattr(obj, key), value)
        else:
            setattr(obj, key, value)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


class EndToEndConfig:
    def __init__(self, warmup, profile, joint, partition, handoff, output_dir):
        self.warmup, self.profile, self.joint = warmup, profile, joint
        self.partition, self.handoff = partition, handoff
        self.output_dir = str(Path(output_dir).resolve())
        self.validate()

    @classmethod
    def load(cls, path, output_dir=None):
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text())
        if set(raw) - {'output_dir', 'common', 'warmup', 'joint', 'partition', 'handoff'}:
            raise ValueError('Unknown end-to-end configuration field')
        stages = []
        for name in ['warmup', 'joint']:
            section = raw[name]
            allowed = {'config', 'inline', 'overrides', 'profile'} if name == 'warmup' else {'config', 'inline', 'overrides'}
            if set(section)-allowed:
                raise ValueError(f'Unknown {name} field')
            if ('config' in section) == ('inline' in section):
                raise ValueError(f'{name} needs exactly one of config or inline')
            if 'inline' in section:
                config = ExperimentConfig()
                update_config(config, section['inline'])
            else:
                config = ExperimentConfig.load(path.parent / section['config'])
            update_config(config, raw.get('common', {}))
            update_config(config, section.get('overrides', {}))
            stages.append(config)
        profile = WarmupProfile.from_dict(raw['warmup']['profile'])
        partition = dict(method='weighted_valley_w1_s2', batch_size=8192, image_views=4)
        partition.update(raw.get('partition', {}))
        handoff = dict(gaussian_count=100000, pca_neighbors=8, reset_density=1e-8)
        handoff.update(raw.get('handoff', {}))
        return cls(stages[0], profile, stages[1], partition, handoff,
                   output_dir or raw['output_dir'])

    def validate(self):
        for c in [self.warmup, self.joint]: c.validate()
        self.profile.validate()
        if self.warmup.representation_mode != 'voronoi_only' or not self.warmup.voronoi_warmup.enabled:
            raise ValueError('Warmup must be Voronoi-only')
        if self.joint.representation_mode != 'hybrid':
            raise ValueError('Joint stage must be hybrid')
        if self.joint.voronoi_adaptation.enabled:
            raise ValueError('End-to-end baseline preserves all retained Voronoi cells; disable joint population adaptation')
        if self.joint.voronoi_warmup.enabled:
            raise ValueError('Joint stage cannot have a second warmup')
        if self.warmup.dataset != self.joint.dataset or self.warmup.background != self.joint.background:
            raise ValueError('Warmup and joint dataset/background must match')
        if self.warmup.seed != self.joint.seed:
            raise ValueError('Warmup and joint seeds must match')
        if set(self.partition) != {'method','batch_size','image_views'}:
            raise ValueError('Unknown partition settings')
        if self.partition['method'] not in {s['name'] for s in variants()}:
            raise ValueError('Unsupported partition method')
        if set(self.handoff) != {'gaussian_count','pca_neighbors','reset_density'}:
            raise ValueError('Unknown handoff settings')
        for mapping, keys in [(self.partition, ['batch_size','image_views']),
                              (self.handoff,['gaussian_count','pca_neighbors'])]:
            for key in keys:
                if not isinstance(mapping[key],int) or isinstance(mapping[key],bool) or mapping[key]<1:
                    raise ValueError(f'{key} must be a positive integer')
        if self.handoff['pca_neighbors'] < 3:
            raise ValueError('pca_neighbors must be at least 3')
        import math
        if not math.isfinite(self.handoff['reset_density']) or self.handoff['reset_density']<=0:
            raise ValueError('reset_density must be finite and positive')
        if self.profile.resolution(self.warmup.optimization.iterations-1) != 1:
            raise ValueError('Warmup must reach base resolution before partitioning')

    def manifest(self):
        return dict(warmup=self.warmup.as_dict(), profile=self.profile.as_dict(),
                    joint=self.joint.as_dict(), partition=self.partition,
                    handoff=self.handoff, output_dir=self.output_dir)


class EndToEndTrainer:
    def __init__(self, config):
        self.config = config

    def fit(self, resume=False):
        """The single public entry point; warmup and initialization are automatic."""
        c = self.config
        root = Path(c.output_dir)
        c.validate()
        manifest = c.manifest()
        if root.exists() and any(root.iterdir()):
            if not resume:
                raise FileExistsError(f'Nonempty output: {root}; use --resume to continue')
            saved = json.loads((root/'pipeline_config.json').read_text())
            for name in ('warmup', 'joint'):
                saved[name] = drop_legacy_keys(saved[name])
            if saved != manifest:
                raise ValueError('Resume requires the same resolved end-to-end configuration')
        else:
            root.mkdir(parents=True, exist_ok=True)
            atomic_json(root/'pipeline_config.json', manifest)
            for name, source in [('warmup', c.warmup), ('joint', c.joint)]:
                stage = deepcopy(source)
                stage.output_dir = str(root/name)
                stage.save(root/f'{name}.yaml')
        for stage in ['warmup','partition','joint']:
            marker = root/f'{stage}_done.json'
            if marker.exists():
                done = json.loads(marker.read_text())
                if not Path(done['artifact']).is_file():
                    raise FileNotFoundError(f'Missing completed-stage artifact: {done["artifact"]}')
                continue
            command = [sys.executable, str(ROOT/'train.py'),
                       '--stage', stage, '--output-dir', str(root)]
            print(f'FOAMSURF_STAGE={stage}', flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            if not marker.is_file():
                raise RuntimeError(f'{stage} did not complete')
        result = json.loads((root/'joint/summary.json').read_text())
        atomic_json(root/'summary.json', dict(status='completed', warmup_iterations=c.warmup.optimization.iterations,
                    joint_iterations=c.joint.optimization.iterations, joint=result))
        return result


def run_stage(root, stage):
    """Internal worker called by fit(); each stage owns its CUDA lifetime."""
    root = Path(root).resolve()
    manifest = json.loads((root/'pipeline_config.json').read_text())
    if stage == 'warmup':
        from training.warmup_trainer import WarmupTrainer
        config = ExperimentConfig.load(root/'warmup.yaml')
        profile = WarmupProfile.from_dict(manifest['profile'])
        last = root/'warmup/checkpoints/last.pt'
        trainer = WarmupTrainer(config, profile, resume=last if last.exists() else None)
        if trainer.start_iteration == config.optimization.iterations + 1:
            # Recover a crash between the final checkpoint and the stage marker.
            result = trainer._write_summary(config.optimization.iterations, 'completed')
            trainer.logger.close()
        else:
            result = trainer.fit()
        if result['status'] != 'completed' or result['final_iteration'] != config.optimization.iterations:
            raise RuntimeError('Warmup stopped before its requested final iteration')
        atomic_json(root/'warmup_done.json', dict(artifact=str(last), iterations=config.optimization.iterations))
    elif stage == 'partition':
        from training.native_handoff import partition_warmup
        artifact = partition_warmup(root, manifest)
        atomic_json(root/'partition_done.json', dict(artifact=str(artifact)))
    elif stage == 'joint':
        import torch
        from training.trainer import Trainer
        from training.native_handoff import initialize_native_hybrid
        config = ExperimentConfig.load(root/'joint.yaml')
        last = root/'joint/checkpoints/last.pt'
        if last.exists():
            trainer = Trainer(config, resume=last)
        else:
            import random
            import numpy as np
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            model, report = initialize_native_hybrid(root, config, manifest['handoff'])
            # Store exact counts for clean checkpoint reload and diagnostics.
            config.save(root/'joint.yaml')
            atomic_json(root/'handoff_report.json', report)
            trainer = Trainer(config, initial_model=model)
            trainer._checkpoint(root/'joint/checkpoints/initialization.pt', 0)
            trainer._checkpoint(last, 0)
            from training.native_handoff import render_transition
            render_transition(trainer, root/'transition', manifest['partition'])
        if trainer.start_iteration == config.optimization.iterations + 1:
            result = trainer._write_summary(config.optimization.iterations, 'completed')
            trainer.logger.close()
        else:
            result = trainer.fit()
        if result['status'] != 'completed' or result['final_iteration'] != config.optimization.iterations:
            raise RuntimeError('Joint stage stopped before its requested final iteration')
        atomic_json(root/'joint_done.json', dict(artifact=str(last), iterations=config.optimization.iterations))
    else:
        raise ValueError(f'Unknown stage: {stage}')
