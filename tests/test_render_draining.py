from pathlib import Path
import subprocess
import sys
import pytest
from scripts.render_draining import commands


def package(tmp_path):
    for name in ['scene-a','scene-b']:
        checkpoint=tmp_path/'models'/name/'joint/checkpoints/last.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        (tmp_path/'data'/name).mkdir(parents=True)
    return tmp_path


def test_pipeline_dependencies_and_preview(tmp_path):
    root=package(tmp_path/'package')
    steps=commands(root,'scene-a',tmp_path/'out',640,'camera',True)
    assert len(steps)==3
    assert '--preview' not in steps[0] and '--preview' not in steps[1]
    assert '--preview' in steps[2] and '--water-surface' in steps[2]
    assert steps[1][steps[1].index('--schedule')+1]==str(tmp_path/'out/calibration/frame_stats.json')
    assert steps[2][steps[2].index('--reference-levels')+1]==str(tmp_path/'out/smooth/frame_stats.json')
    assert all(step[step.index('--view')+1]=='camera' for step in steps)
    with pytest.raises(ValueError,match='Missing'):
        commands(root,'missing',tmp_path/'out')


def test_all_dry_run_works_outside_repository(tmp_path):
    root=package(tmp_path/'package')
    script=Path(__file__).resolve().parents[1]/'scripts/render_draining.py'
    out=tmp_path/'renders'
    result=subprocess.run([sys.executable,str(script),'--package',str(root),'--all','--out',str(out),'--dry-run'],cwd=tmp_path,text=True,capture_output=True,check=True)
    assert result.stdout.count('stage ')==6
    assert not out.exists()


def test_breaking_effect_only_runs_in_final_stage(tmp_path):
    root=package(tmp_path/'package')
    steps=commands(root,'scene-a',tmp_path/'out',breaking=True)
    assert '--breaking-waves' not in steps[0]
    assert '--breaking-waves' not in steps[1]
    assert '--breaking-waves' in steps[2]


def test_strong_effect_and_extra_final_hold_are_forwarded(tmp_path):
    root=package(tmp_path/'package')
    steps=commands(root,'scene-a',tmp_path/'out',breaking=True,impact_strength=4,end_hold=3)
    last=steps[-1]
    assert last[last.index('--impact-strength')+1]=='4'
    assert last[last.index('--end-hold')+1]=='3'


def test_surface_dynamics_only_in_final_stage(tmp_path):
    root=package(tmp_path/'package')
    steps=commands(root,'scene-a',tmp_path/'out',dynamics=True)
    assert all('--surface-dynamics' not in step for step in steps[:2])
    assert '--surface-dynamics' in steps[-1] and '--breaking-waves' in steps[-1]


def test_final_wave_scale_is_forwarded(tmp_path):
    root=package(tmp_path/'package')
    last=commands(root,'scene-a',tmp_path/'out',dynamics=True,final_wave_scale=2)[-1]
    assert last[last.index('--final-wave-scale')+1]=='2'


def test_final_rise_only_applies_to_final_animation(tmp_path):
    root=package(tmp_path/'package')
    steps=commands(root,'scene-a',tmp_path/'out',dynamics=True,final_rise=.1,rise_seconds=5)
    assert all('--final-rise' not in step for step in steps[:2])
    assert steps[-1][steps[-1].index('--final-rise')+1]=='0.1'
    assert steps[-1][steps[-1].index('--rise-seconds')+1]=='5'


def test_post_rise_hold_only_forwarded_to_final_animation(tmp_path):
    root=package(tmp_path/'package')
    steps=commands(root,'scene-a',tmp_path/'out',dynamics=True,post_rise_hold=10)
    assert all('--post-rise-hold' not in step for step in steps[:2])
    assert steps[-1][steps[-1].index('--post-rise-hold')+1]=='10'
