PPISP v1.0.0 (https://github.com/nv-tlabs/ppisp, commit 86c2c64), Apache-2.0: see LICENSE and
ATTRIBUTIONS.md. Deutsch et al., "PPISP: Physically-Plausible Compensation and Control of Photometric
Variations in Radiance Field Reconstruction".

The `ppisp` package is used unchanged. Its CUDA extension `ppisp_cuda` is compiled at first use by
training/camera_isp.py (torch JIT, TORCH_CUDA_ARCH_LIST as for the other extensions) instead of setup.py.
