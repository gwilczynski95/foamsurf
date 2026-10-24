"""scripts/slim_run.py and the space-saving defaults of train_seathru.sbatch."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import slim_run  # noqa: E402


def fake_run(root, psnr_by_iteration, finished=True):
    joint = root / "joint"
    (joint / "metrics").mkdir(parents=True)
    rows = "\n".join(f"{i},{p}" for i, p in psnr_by_iteration.items())
    (joint / "metrics/eval.csv").write_text("iteration,eval/hybrid_psnr\n" + rows + "\n")
    for i in psnr_by_iteration:
        folder = joint / "evaluation" / f"iteration_{i:06d}"
        folder.mkdir(parents=True)
        (folder / "0000_A_hybrid.png").write_bytes(b"x" * 10)
    (joint / "checkpoints").mkdir()
    for name in ("best.pt", "last.pt", "initialization.pt", "best.smooth.json"):
        (joint / "checkpoints" / name).write_bytes(b"c")
    (joint / "wandb/latest-run/files").mkdir(parents=True)
    (joint / "wandb/latest-run/files/media.png").write_bytes(b"w" * 100)
    if finished:
        (joint / "summary.json").write_text("{}")
    return root


def test_keeps_every_5000_last_and_best(tmp_path):
    run = fake_run(tmp_path / "r", {0: 10.0, 500: 11.0, 5000: 20.0, 7500: 29.0, 10000: 25.0, 10500: 24.0})
    assert slim_run.kept_iterations(run) == {0, 5000, 7500, 10000, 10500}
    removed = {p.name for p in slim_run.removable(run)}
    assert removed == {"wandb", "initialization.pt", "iteration_000500"}
    slim_run.remove(slim_run.removable(run))
    left = sorted(p.name for p in (run / "joint/evaluation").iterdir())
    assert left == ["iteration_000000", "iteration_005000", "iteration_007500", "iteration_010000", "iteration_010500"]
    assert {p.name for p in (run / "joint/checkpoints").iterdir()} == {"best.pt", "last.pt", "best.smooth.json"}
    assert not (run / "joint/wandb").exists() and (run / "joint/metrics/eval.csv").is_file()


def test_refuses_an_unfinished_run_and_dry_run_deletes_nothing(tmp_path):
    run = fake_run(tmp_path / "r", {0: 10.0, 500: 11.0, 1000: 12.0}, finished=False)
    script = [sys.executable, str(ROOT / "scripts/slim_run.py"), "--run", str(run)]
    result = subprocess.run(script + ["--apply"], capture_output=True, text=True)
    assert result.returncode != 0 and (run / "joint/evaluation/iteration_000500").is_dir()
    (run / "joint/summary.json").write_text("{}")
    result = subprocess.run(script, capture_output=True, text=True, check=True)
    assert "would be freed" in result.stdout and (run / "joint/evaluation/iteration_000500").is_dir()


def test_sbatch_disables_wandb_and_slims_by_default():
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    assert 'WANDB="${FS_WANDB:-disabled}"' in script
    assert '--set "warmup.wandb.mode=$WANDB" --set "joint.wandb.mode=$WANDB"' in script
    assert 'python scripts/slim_run.py --run "$OUTPUT" --keep-until "${FS_SLIM_KEEP_UNTIL:-0}" --apply' in script


def test_keep_until_keeps_the_start_frame_by_frame(tmp_path):
    run = fake_run(tmp_path / "r", {i: 10.0 + i / 1e4 for i in range(0, 8001, 500)})
    keep = slim_run.kept_iterations(run, keep_until=5000)
    assert keep == set(range(0, 5001, 500)) | {8000}  # every 500 up to 5000, then last (= best)
    assert slim_run.kept_iterations(run) == {0, 5000, 8000}
