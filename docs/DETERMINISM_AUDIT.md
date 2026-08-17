# FoamSurf determinism audit

The audit separates exact numerical determinism from experimental stability.
All runs are logged to the W&B project `foamsurf_determinism_audit` and write
raw JSON/CSV/checkpoints below `outputs/foamsurf_determinism_audit`.

## Replay audit

`determinism_audit_replay.sbatch` creates the same initialization twice and
replays the same full-image forward, backward and topology proposal five times
inside each process. It records tensor hashes, maximum differences, candidate
mask XOR counts, clone/split/prune index hashes and the number of gradients
within 1%, 5% and 10% of the densification threshold.

Three array tasks run sequentially so they reuse one node/GPU rather than
measuring concurrent-device differences:

```bash
sbatch --nodelist=c13 scripts/determinism_audit_replay.sbatch
```

Repeat on the node which produced the earlier successful sweep1/05 run:

```bash
sbatch --nodelist=c17 scripts/determinism_audit_replay.sbatch
```

## Replicated training audit

Tasks 0-2 train 1,000 steps with Gaussian adaptation disabled. Tasks 3-5 run
the identical seed-42 `05_reset005` control through 12,000 steps with
adaptation enabled. Tasks are sequential on the allocated node.

```bash
sbatch --nodelist=c13 scripts/determinism_audit_training.sbatch
sbatch --nodelist=c17 scripts/determinism_audit_training.sbatch
```

The first group measures whether ordinary optimization diverges before any
discrete topology decision. The second measures topology amplification and
captures whether the same configuration completes 12k or reaches the one
million Gaussian safety threshold.

## Provenance

Every task stores `provenance.json` containing the node, GPU, Python, PyTorch,
CUDA, Git state and SHA-256 hashes of both native renderer binaries. This is
also attached to the W&B run configuration.

The audit explicitly reports that current checkpoints omit Python's RNG,
NumPy's RNG and the camera-without-replacement stack. This means resume
equivalence is not expected until checkpoint state is extended; it is a known
issue discovered by the audit, not silently treated as renderer noise.

## Acceptance criteria

Before trusting another ablation sweep, repeated same-seed runs should have:

- identical initialization and camera sequence;
- exact or tightly bounded forward/backward replay differences;
- zero (ideally) or negligible candidate-mask XOR at topology events;
- no completed-versus-runaway status disagreement;
- population trajectories close at every topology update;
- final metric variation reported over multiple seeds after the deterministic
  same-seed audit passes.
