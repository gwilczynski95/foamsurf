#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include "hybrid_cuda.h"

namespace py = pybind11;

namespace foamsurf_bindings {
namespace {

void require_float_cuda(const torch::Tensor &value, const char *name) {
    TORCH_CHECK(value.is_cuda(), name, " must be on CUDA");
    TORCH_CHECK(value.scalar_type() == torch::kFloat32,
                name,
                " must have float32 dtype");
}

void validate_inputs(const torch::Tensor &points,
                     const torch::Tensor &densities,
                     const torch::Tensor &medium_colors,
                     const torch::Tensor &adjacency,
                     const torch::Tensor &offsets,
                     const torch::Tensor &rays,
                     const torch::Tensor &start_points,
                     const torch::Tensor &hit_depths,
                     const torch::Tensor &hit_indices,
                     const torch::Tensor &means,
                     const torch::Tensor &log_scales,
                     const torch::Tensor &quaternions,
                     const torch::Tensor &gaussian_colors,
                     const torch::Tensor &opacity_logits,
                     const torch::Tensor &raw_kappa,
                     const torch::Tensor &background,
                     int64_t density_columns = 1) {
    require_float_cuda(points, "points");
    require_float_cuda(densities, "densities");
    require_float_cuda(medium_colors, "medium_colors");
    require_float_cuda(rays, "rays");
    require_float_cuda(hit_depths, "hit_depths");
    require_float_cuda(means, "means");
    require_float_cuda(log_scales, "log_scales");
    require_float_cuda(quaternions, "quaternions");
    require_float_cuda(gaussian_colors, "gaussian_colors");
    require_float_cuda(opacity_logits, "opacity_logits");
    require_float_cuda(raw_kappa, "raw_kappa");
    require_float_cuda(background, "background");
    TORCH_CHECK(adjacency.is_cuda() && adjacency.scalar_type() == torch::kUInt32,
                "adjacency must be CUDA uint32");
    TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == torch::kUInt32,
                "offsets must be CUDA uint32");
    TORCH_CHECK(start_points.is_cuda() &&
                    start_points.scalar_type() == torch::kUInt32,
                "start_points must be CUDA uint32");
    TORCH_CHECK(hit_indices.is_cuda() &&
                    hit_indices.scalar_type() == torch::kInt32,
                "hit_indices must be CUDA int32");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
                "points must have shape [V, 3]");
    TORCH_CHECK(densities.dim() == 2 && densities.size(0) == points.size(0) &&
                    densities.size(1) == density_columns,
                "densities must have shape [V, ", density_columns, "]");
    TORCH_CHECK(medium_colors.dim() == 2 &&
                    medium_colors.size(0) == points.size(0) &&
                    medium_colors.size(1) == 3,
                "medium_colors must have shape [V, 3]");
    TORCH_CHECK(offsets.numel() == points.size(0) + 1,
                "offsets must contain V + 1 elements");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 6,
                "rays must have shape [R, 6]");
    TORCH_CHECK(start_points.numel() == rays.size(0),
                "start_points must contain one index per ray");
    TORCH_CHECK(hit_depths.dim() == 2 && hit_indices.dim() == 2 &&
                    hit_depths.size(0) == hit_indices.size(0) &&
                    hit_depths.size(1) == hit_indices.size(1) &&
                    hit_depths.size(1) == rays.size(0),
                "hit buffers must have matching [K, R] shapes");
    int64_t gaussian_count = means.size(0);
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3,
                "means must have shape [G, 3]");
    TORCH_CHECK(log_scales.dim() == 2 &&
                    log_scales.size(0) == gaussian_count &&
                    log_scales.size(1) == 2,
                "log_scales must have shape [G, 2]");
    TORCH_CHECK(quaternions.dim() == 2 &&
                    quaternions.size(0) == gaussian_count &&
                    quaternions.size(1) == 4,
                "quaternions must have shape [G, 4]");
    TORCH_CHECK(gaussian_colors.dim() == 2 &&
                    gaussian_colors.size(0) == gaussian_count &&
                    gaussian_colors.size(1) == 3,
                "gaussian_colors must have shape [G, 3]");
    TORCH_CHECK(opacity_logits.numel() == gaussian_count,
                "opacity_logits must have G values");
    TORCH_CHECK(raw_kappa.numel() == gaussian_count,
                "raw_kappa must have G values");
    TORCH_CHECK(background.numel() == 3,
                "background must contain three values");
}

foamsurf::HybridLaunchSettings settings_for(const torch::Tensor &points,
                                             const torch::Tensor &means,
                                             const torch::Tensor &rays,
                                             const torch::Tensor &hit_depths,
                                             float near_plane,
                                             float far_plane,
                                             float threshold,
                                             uint32_t max_intersections) {
    TORCH_CHECK(far_plane > near_plane, "far_plane must exceed near_plane");
    return {static_cast<uint32_t>(points.size(0)),
            static_cast<uint32_t>(means.size(0)),
            static_cast<uint32_t>(rays.size(0)),
            static_cast<uint32_t>(hit_depths.size(0)),
            max_intersections,
            near_plane,
            far_plane,
            threshold};
}

using TensorList = std::vector<torch::Tensor>;

py::tuple hybrid_forward(TensorList values,
                         float near_plane,
                         float far_plane,
                         float threshold,
                         uint32_t max_intersections) {
    TORCH_CHECK(values.size() == 16, "hybrid_forward expects 16 tensors");
    for (auto &value : values) {
        value = value.contiguous();
    }
    validate_inputs(values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    values[6],
                    values[7],
                    values[8],
                    values[9],
                    values[10],
                    values[11],
                    values[12],
                    values[13],
                    values[14],
                    values[15]);
    auto settings = settings_for(values[0],
                                 values[9],
                                 values[5],
                                 values[7],
                                 near_plane,
                                 far_plane,
                                 threshold,
                                 max_intersections);
    auto output = torch::empty({values[5].size(0), 4}, values[5].options());
    auto intersections = torch::empty(
        {values[5].size(0), 1}, values[5].options().dtype(torch::kUInt32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    foamsurf::hybrid_forward_cuda(settings,
                                  values[0].data_ptr<float>(),
                                  values[1].data_ptr<float>(),
                                  values[2].data_ptr<float>(),
                                  values[3].data_ptr<uint32_t>(),
                                  values[4].data_ptr<uint32_t>(),
                                  values[5].data_ptr<float>(),
                                  values[6].data_ptr<uint32_t>(),
                                  values[7].data_ptr<float>(),
                                  values[8].data_ptr<int32_t>(),
                                  values[9].data_ptr<float>(),
                                  values[10].data_ptr<float>(),
                                  values[11].data_ptr<float>(),
                                  values[12].data_ptr<float>(),
                                  values[13].data_ptr<float>(),
                                  values[14].data_ptr<float>(),
                                  values[15].data_ptr<float>(),
                                  output.data_ptr<float>(),
                                  intersections.data_ptr<uint32_t>(),
                                  &stream);
    return py::make_tuple(output, intersections);
}

py::tuple hybrid_backward(TensorList values,
                          torch::Tensor output,
                          torch::Tensor output_grad,
                          float near_plane,
                          float far_plane,
                          float threshold,
                          uint32_t max_intersections) {
    TORCH_CHECK(values.size() == 16, "hybrid_backward expects 16 tensors");
    for (auto &value : values) {
        value = value.contiguous();
    }
    output = output.contiguous();
    output_grad = output_grad.contiguous();
    validate_inputs(values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    values[6],
                    values[7],
                    values[8],
                    values[9],
                    values[10],
                    values[11],
                    values[12],
                    values[13],
                    values[14],
                    values[15]);
    require_float_cuda(output, "output");
    require_float_cuda(output_grad, "output_grad");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == values[5].size(0) &&
                    output.size(1) == 4 && output_grad.sizes() == output.sizes(),
                "output and output_grad must have shape [R, 4]");
    auto settings = settings_for(values[0],
                                 values[9],
                                 values[5],
                                 values[7],
                                 near_plane,
                                 far_plane,
                                 threshold,
                                 max_intersections);
    TensorList gradients = {torch::zeros_like(values[0]),
                            torch::zeros_like(values[1]),
                            torch::zeros_like(values[2]),
                            torch::zeros_like(values[9]),
                            torch::zeros_like(values[10]),
                            torch::zeros_like(values[11]),
                            torch::zeros_like(values[12]),
                            torch::zeros_like(values[13]),
                            torch::zeros_like(values[14])};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    foamsurf::hybrid_backward_cuda(settings,
                                   values[0].data_ptr<float>(),
                                   values[1].data_ptr<float>(),
                                   values[2].data_ptr<float>(),
                                   values[3].data_ptr<uint32_t>(),
                                   values[4].data_ptr<uint32_t>(),
                                   values[5].data_ptr<float>(),
                                   values[6].data_ptr<uint32_t>(),
                                   values[7].data_ptr<float>(),
                                   values[8].data_ptr<int32_t>(),
                                   values[9].data_ptr<float>(),
                                   values[10].data_ptr<float>(),
                                   values[11].data_ptr<float>(),
                                   values[12].data_ptr<float>(),
                                   values[13].data_ptr<float>(),
                                   values[14].data_ptr<float>(),
                                   values[15].data_ptr<float>(),
                                   output.data_ptr<float>(),
                                   output_grad.data_ptr<float>(),
                                   gradients[0].data_ptr<float>(),
                                   gradients[1].data_ptr<float>(),
                                   gradients[2].data_ptr<float>(),
                                   gradients[3].data_ptr<float>(),
                                   gradients[4].data_ptr<float>(),
                                   gradients[5].data_ptr<float>(),
                                   gradients[6].data_ptr<float>(),
                                   gradients[7].data_ptr<float>(),
                                   gradients[8].data_ptr<float>(),
                                   &stream);
    py::tuple result(gradients.size());
    for (size_t index = 0; index < gradients.size(); ++index) {
        result[index] = gradients[index];
    }
    return result;
}

py::tuple hybrid_diagnostics(TensorList values,
                             float near_plane,
                             float far_plane,
                             float threshold,
                             uint32_t max_intersections) {
    TORCH_CHECK(values.size() == 17, "hybrid_diagnostics expects 17 tensors");
    for (auto &value : values) {
        value = value.contiguous();
    }
    validate_inputs(values[0], values[1], values[2], values[3], values[4],
                    values[5], values[6], values[7], values[8], values[9],
                    values[10], values[11], values[12], values[13], values[14],
                    values[15]);
    require_float_cuda(values[16], "ray_errors");
    TORCH_CHECK(values[16].numel() == values[5].size(0),
                "ray_errors must contain one value per ray");
    auto settings = settings_for(values[0], values[9], values[5], values[7],
                                 near_plane, far_plane, threshold,
                                 max_intersections);
    auto ray_statistics =
        torch::zeros({values[5].size(0), 8}, values[5].options());
    auto gaussian_statistics =
        torch::zeros({values[9].size(0), 4}, values[5].options());
    auto medium_statistics =
        torch::zeros({values[0].size(0), 5}, values[5].options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    foamsurf::hybrid_diagnostics_cuda(
        settings, values[0].data_ptr<float>(), values[1].data_ptr<float>(),
        values[3].data_ptr<uint32_t>(), values[4].data_ptr<uint32_t>(),
        values[5].data_ptr<float>(), values[6].data_ptr<uint32_t>(),
        values[7].data_ptr<float>(), values[8].data_ptr<int32_t>(),
        values[9].data_ptr<float>(), values[10].data_ptr<float>(),
        values[11].data_ptr<float>(), values[13].data_ptr<float>(),
        values[14].data_ptr<float>(), values[16].data_ptr<float>(),
        ray_statistics.data_ptr<float>(),
        gaussian_statistics.data_ptr<float>(),
        medium_statistics.data_ptr<float>(),
        &stream);
    return py::make_tuple(
        ray_statistics, gaussian_statistics, medium_statistics);
}

py::tuple surface_hits(torch::Tensor hit_depths,
                       torch::Tensor hit_indices,
                       float near_plane,
                       float far_plane) {
    hit_depths = hit_depths.contiguous();
    hit_indices = hit_indices.contiguous();
    require_float_cuda(hit_depths, "hit_depths");
    TORCH_CHECK(hit_indices.is_cuda() &&
                    hit_indices.scalar_type() == torch::kInt32,
                "hit_indices must be CUDA int32");
    TORCH_CHECK(hit_depths.dim() == 2 && hit_depths.size(0) > 0 &&
                    hit_indices.sizes() == hit_depths.sizes(),
                "hit buffers must have matching [K, R] shapes with K > 0");
    auto ray_count = hit_depths.size(1);
    auto depth = torch::empty({ray_count}, hit_depths.options());
    auto index = torch::empty({ray_count}, hit_indices.options());
    if (ray_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        foamsurf::surface_hit_cuda(static_cast<uint32_t>(ray_count),
                                   static_cast<uint32_t>(hit_depths.size(0)),
                                   near_plane,
                                   far_plane,
                                   hit_depths.data_ptr<float>(),
                                   hit_indices.data_ptr<int32_t>(),
                                   depth.data_ptr<float>(),
                                   index.data_ptr<int32_t>(),
                                   &stream);
    }
    return py::make_tuple(depth, index);
}

// RGB tensor list: points, attenuation [V,3], backscatter [V,3], medium_colors,
// adjacency, offsets, rays, start_points, hit_depths, hit_indices, means,
// log_scales, quaternions, gaussian_colors, opacity_logits, raw_kappa,
// background.
void validate_rgb_inputs(TensorList &values) {
    TORCH_CHECK(values.size() == 17, "RGB medium expects 17 tensors");
    for (auto &value : values) {
        value = value.contiguous();
    }
    validate_inputs(values[0], values[1], values[3], values[4], values[5],
                    values[6], values[7], values[8], values[9], values[10],
                    values[11], values[12], values[13], values[14], values[15],
                    values[16], 3);
    require_float_cuda(values[2], "backscatter");
    TORCH_CHECK(values[2].sizes() == values[1].sizes(),
                "backscatter must have the shape of attenuation");
}

foamsurf::RgbSceneTensors rgb_scene(const TensorList &values) {
    return {values[0].data_ptr<float>(),
            values[1].data_ptr<float>(),
            values[2].data_ptr<float>(),
            values[3].data_ptr<float>(),
            values[4].data_ptr<uint32_t>(),
            values[5].data_ptr<uint32_t>(),
            values[6].data_ptr<float>(),
            values[7].data_ptr<uint32_t>(),
            values[8].data_ptr<float>(),
            values[9].data_ptr<int32_t>(),
            values[10].data_ptr<float>(),
            values[11].data_ptr<float>(),
            values[12].data_ptr<float>(),
            values[13].data_ptr<float>(),
            values[14].data_ptr<float>(),
            values[15].data_ptr<float>()};
}

py::tuple rgb_forward(TensorList values,
                      float near_plane,
                      float far_plane,
                      float threshold,
                      uint32_t max_intersections) {
    validate_rgb_inputs(values);
    auto settings = settings_for(values[0], values[10], values[6], values[8],
                                 near_plane, far_plane, threshold,
                                 max_intersections);
    auto ray_count = values[6].size(0);
    auto direct = torch::empty({ray_count, 3}, values[6].options());
    auto backscatter = torch::empty({ray_count, 3}, values[6].options());
    auto transmission = torch::empty({ray_count, 3}, values[6].options());
    auto intersections = torch::empty(
        {ray_count, 1}, values[6].options().dtype(torch::kUInt32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    foamsurf::rgb_forward_cuda(settings,
                               rgb_scene(values),
                               values[16].data_ptr<float>(),
                               direct.data_ptr<float>(),
                               backscatter.data_ptr<float>(),
                               transmission.data_ptr<float>(),
                               intersections.data_ptr<uint32_t>(),
                               &stream);
    return py::make_tuple(direct, backscatter, transmission, intersections);
}

py::tuple rgb_backward(TensorList values,
                       TensorList outputs,
                       TensorList output_grads,
                       float near_plane,
                       float far_plane,
                       float threshold,
                       uint32_t max_intersections,
                       c10::optional<torch::Tensor> screen_axes,
                       c10::optional<torch::Tensor> means_screen_abs) {
    validate_rgb_inputs(values);
    TORCH_CHECK(screen_axes.has_value() == means_screen_abs.has_value(),
                "screen_axes and means_screen_abs go together");
    TORCH_CHECK(outputs.size() == 3 && output_grads.size() == 3,
                "outputs and output_grads must be (direct, backscatter, "
                "transmission)");
    auto ray_count = values[6].size(0);
    for (size_t index = 0; index < 3; ++index) {
        outputs[index] = outputs[index].contiguous();
        output_grads[index] = output_grads[index].contiguous();
        require_float_cuda(outputs[index], "output");
        require_float_cuda(output_grads[index], "output_grad");
        TORCH_CHECK(outputs[index].dim() == 2 &&
                        outputs[index].size(0) == ray_count &&
                        outputs[index].size(1) == 3 &&
                        output_grads[index].sizes() == outputs[index].sizes(),
                    "outputs and output_grads must have shape [R, 3]");
    }
    auto settings = settings_for(values[0], values[10], values[6], values[8],
                                 near_plane, far_plane, threshold,
                                 max_intersections);
    TensorList gradients = {torch::zeros_like(values[0]),
                            torch::zeros_like(values[1]),
                            torch::zeros_like(values[2]),
                            torch::zeros_like(values[3]),
                            torch::zeros_like(values[10]),
                            torch::zeros_like(values[11]),
                            torch::zeros_like(values[12]),
                            torch::zeros_like(values[13]),
                            torch::zeros_like(values[14]),
                            torch::zeros_like(values[15])};
    foamsurf::RgbGradientTensors gradient_pointers{
        gradients[0].data_ptr<float>(), gradients[1].data_ptr<float>(),
        gradients[2].data_ptr<float>(), gradients[3].data_ptr<float>(),
        gradients[4].data_ptr<float>(), gradients[5].data_ptr<float>(),
        gradients[6].data_ptr<float>(), gradients[7].data_ptr<float>(),
        gradients[8].data_ptr<float>(), gradients[9].data_ptr<float>()};
    if (means_screen_abs.has_value()) {
        // Written in place: the caller owns and zeroes the [N,2] buffer.
        require_float_cuda(*screen_axes, "screen_axes");
        require_float_cuda(*means_screen_abs, "means_screen_abs");
        TORCH_CHECK(screen_axes->is_contiguous() && screen_axes->numel() == 6,
                    "screen_axes must be a contiguous [2,3] tensor");
        TORCH_CHECK(means_screen_abs->is_contiguous() &&
                        means_screen_abs->dim() == 2 &&
                        means_screen_abs->size(0) == values[10].size(0) &&
                        means_screen_abs->size(1) == 2,
                    "means_screen_abs must be a contiguous [N,2] tensor");
        gradient_pointers.screen_axes = screen_axes->data_ptr<float>();
        gradient_pointers.means_screen_abs = means_screen_abs->data_ptr<float>();
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    foamsurf::rgb_backward_cuda(settings,
                                rgb_scene(values),
                                outputs[0].data_ptr<float>(),
                                outputs[1].data_ptr<float>(),
                                outputs[2].data_ptr<float>(),
                                output_grads[0].data_ptr<float>(),
                                output_grads[1].data_ptr<float>(),
                                output_grads[2].data_ptr<float>(),
                                gradient_pointers,
                                &stream);
    py::tuple result(gradients.size());
    for (size_t index = 0; index < gradients.size(); ++index) {
        result[index] = gradients[index];
    }
    return result;
}

} // namespace

void init_hybrid_bindings(py::module &module) {
    module.def("hybrid_forward",
               hybrid_forward,
               py::arg("tensors"),
               py::arg("near_plane"),
               py::arg("far_plane"),
               py::arg("threshold") = 0.001f,
               py::arg("max_intersections") = 1024);
    module.def("hybrid_backward",
               hybrid_backward,
               py::arg("tensors"),
               py::arg("output"),
               py::arg("output_grad"),
               py::arg("near_plane"),
               py::arg("far_plane"),
               py::arg("threshold") = 0.001f,
               py::arg("max_intersections") = 1024);
    module.def("hybrid_diagnostics",
               hybrid_diagnostics,
               py::arg("tensors"),
               py::arg("near_plane"),
               py::arg("far_plane"),
               py::arg("threshold") = 0.001f,
               py::arg("max_intersections") = 1024);
    module.def("surface_hits",
               surface_hits,
               py::arg("hit_depths"),
               py::arg("hit_indices"),
               py::arg("near_plane"),
               py::arg("far_plane"));
    module.def("rgb_forward",
               rgb_forward,
               py::arg("tensors"),
               py::arg("near_plane"),
               py::arg("far_plane"),
               py::arg("threshold") = 0.001f,
               py::arg("max_intersections") = 1024);
    module.def("rgb_backward",
               rgb_backward,
               py::arg("tensors"),
               py::arg("outputs"),
               py::arg("output_grads"),
               py::arg("near_plane"),
               py::arg("far_plane"),
               py::arg("threshold") = 0.001f,
               py::arg("max_intersections") = 1024,
               py::arg("screen_axes") = py::none(),
               py::arg("means_screen_abs") = py::none());
}

} // namespace foamsurf_bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Standalone FoamSurf hybrid CUDA renderer bindings";
    foamsurf_bindings::init_hybrid_bindings(module);
}
