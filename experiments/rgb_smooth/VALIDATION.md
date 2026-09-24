# Validation, 2026-09-24

New smoothing-only math and integration tests: **10 passed** in the first
targeted run. The full gate then executed **133 tests: 132 passed, 1 failed**.
All 11 new tests passed, including the full-resolution RGB smoothing test.

The failing test was the unchanged legacy
`tests/end_to_end/test_pipeline_gpu.py::test_real_native_pipeline_and_resume`.
It reported CUDA out-of-memory while constructing Delaunay topology in the
joint stage. During the gate, interactive allocation **512889 expired**
(`TIMEOUT`, four-hour limit, end 2026-09-24 13:51:13). Physical GPU 2 was
subsequently assigned to another user's job, occupying about 16.5 GiB.
No further GPU testing was attempted after discovering the expired allocation.
This is not a successful full gate; do not treat it as certification.

Log: `outputs/rgb_smooth_gate_512889.log`.
Aggregate JUnit: `outputs/rgb_smooth_tests/local_512889/passed.xml`.
No new passing certificate was produced. Rerun `test.sbatch` on a dedicated
SLURM allocation, then launch `train.sbatch` using the `afterok` dependency
and certificate commands in README.md. The training sweep was not submitted.

Both new sbatch scripts pass `bash -n`. The old RGB sweep certificate was
checked again after all implementation changes and remains valid, with source
fingerprint `8b93ac2297230225fc415926b71501ea6aa2050ceecf5732f7880b290c1dd56d`.
All changes for this task live under `experiments/rgb_smooth/`; no original
training, configuration, renderer or gate files were modified by this task.
