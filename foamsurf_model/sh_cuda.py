"""View-dependent Gaussian colours in one CUDA kernel, for rendering without gradients (evaluation, viewing).
Same basis and coefficient order as foamsurf_model/sh.py (3DGS convention): colour = activation(sum_k
coefficient_k * basis_k(direction to the camera)), activation sigmoid or 0.5 + value. The Python path
launches dozens of elementwise kernels over all Gaussians per view; this is one launch reading the DC and
rest coefficients in place (no concatenation). Compiled at first use (torch JIT, like the other extensions).
"""
import torch

_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sh_colors_kernel(const float* __restrict__ means, const float* __restrict__ dc,
                                 const float* __restrict__ rest, int rest_count, int degree,
                                 float ox, float oy, float oz, bool sigmoid, int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = ox - means[3 * i], y = oy - means[3 * i + 1], z = oz - means[3 * i + 2];
    const float norm = fmaxf(sqrtf(x * x + y * y + z * z), 1e-12f);  // F.normalize
    x /= norm; y /= norm; z /= norm;
    float basis[16];
    basis[0] = 0.28209479177387814f;
    if (degree > 0) {
        basis[1] = -0.4886025119029199f * y; basis[2] = 0.4886025119029199f * z; basis[3] = -0.4886025119029199f * x;
    }
    if (degree > 1) {
        const float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
        basis[4] = 1.0925484305920792f * xy;
        basis[5] = -1.0925484305920792f * yz;
        basis[6] = 0.31539156525252005f * (2.0f * zz - xx - yy);
        basis[7] = -1.0925484305920792f * xz;
        basis[8] = 0.5462742152960396f * (xx - yy);
        if (degree > 2) {
            basis[9] = -0.5900435899266435f * y * (3.0f * xx - yy);
            basis[10] = 2.890611442640554f * xy * z;
            basis[11] = -0.4570457994644658f * y * (4.0f * zz - xx - yy);
            basis[12] = 0.3731763325901154f * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
            basis[13] = -0.4570457994644658f * x * (4.0f * zz - xx - yy);
            basis[14] = 1.445305721320277f * z * (xx - yy);
            basis[15] = -0.5900435899266435f * x * (xx - 3.0f * yy);
        }
    }
    const int count = (degree + 1) * (degree + 1);
    for (int c = 0; c < 3; ++c) {
        float value = dc[3 * i + c] * basis[0];
        const float* r = rest + (size_t)(3 * i + c) * rest_count;
        for (int k = 1; k < count; ++k) value += r[k - 1] * basis[k];
        out[3 * i + c] = sigmoid ? 1.0f / (1.0f + expf(-value)) : 0.5f + value;
    }
}

torch::Tensor sh_colors(torch::Tensor means, torch::Tensor dc, torch::Tensor rest, int64_t degree,
                        torch::Tensor origin, bool sigmoid) {
    TORCH_CHECK(means.is_cuda() && means.is_contiguous() && means.dtype() == torch::kFloat32, "means");
    TORCH_CHECK(dc.is_contiguous() && rest.is_contiguous(), "contiguous coefficients");
    const int n = means.size(0);
    const int rest_count = rest.size(2);
    TORCH_CHECK((degree + 1) * (degree + 1) - 1 <= rest_count, "not enough SH coefficients");
    auto o = origin.to(torch::kCPU, torch::kFloat32).contiguous();
    const float* op = o.data_ptr<float>();
    auto out = torch::empty({n, 3}, means.options());
    if (n == 0) return out;
    const int threads = 256;
    sh_colors_kernel<<<(n + threads - 1) / threads, threads>>>(
        means.data_ptr<float>(), dc.data_ptr<float>(), rest.data_ptr<float>(), rest_count, (int)degree,
        op[0], op[1], op[2], sigmoid, n, out.data_ptr<float>());
    return out;
}
"""

_module = None


def _load():
    global _module
    if _module is None:
        from torch.utils.cpp_extension import load_inline
        _module = load_inline(name="foamsurf_sh_cuda", cpp_sources="torch::Tensor sh_colors(torch::Tensor means, "
                              "torch::Tensor dc, torch::Tensor rest, int64_t degree, torch::Tensor origin, bool sigmoid);",
                              cuda_sources=_SOURCE, functions=["sh_colors"], extra_cuda_cflags=["-O3"], verbose=False)
    return _module


def sh_colors(means, dc, rest, degree, origin, sigmoid):
    """Colours [N, 3] of Gaussians at means [N, 3] with SH coefficients dc [N, 3, 1] and rest [N, 3, R],
    seen from origin [3]; no gradients."""
    if degree < 0 or degree > 3:
        raise ValueError("SH degree must be between 0 and 3")
    return _load().sh_colors(means.detach().contiguous(), dc.detach().contiguous(), rest.detach().contiguous(),
                             int(degree), origin.detach().reshape(3), bool(sigmoid))
