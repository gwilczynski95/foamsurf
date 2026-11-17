#pragma once

#include <cstdint>

namespace foamsurf {

struct HybridLaunchSettings {
    uint32_t num_points;
    uint32_t num_gaussians;
    uint32_t num_rays;
    uint32_t hit_buffer_size;
    uint32_t max_intersections;
    float near_plane;
    float far_plane;
    float transmittance_threshold;
};

void hybrid_forward_cuda(
    const HybridLaunchSettings &settings,
    const float *points,
    const float *densities,
    const float *medium_colors,
    const uint32_t *point_adjacency,
    const uint32_t *point_adjacency_offsets,
    const float *rays,
    const uint32_t *start_points,
    const float *hit_depths,
    const int32_t *hit_indices,
    const float *means,
    const float *log_scales,
    const float *quaternions,
    const float *gaussian_colors,
    const float *opacity_logits,
    const float *raw_kappa,
    const float *background,
    float *output_rgba,
    uint32_t *output_num_intersections,
    const void *stream);

void hybrid_diagnostics_cuda(
    const HybridLaunchSettings &settings,
    const float *points,
    const float *densities,
    const uint32_t *point_adjacency,
    const uint32_t *point_adjacency_offsets,
    const float *rays,
    const uint32_t *start_points,
    const float *hit_depths,
    const int32_t *hit_indices,
    const float *means,
    const float *log_scales,
    const float *quaternions,
    const float *opacity_logits,
    const float *raw_kappa,
    const float *ray_errors,
    float *ray_statistics,
    float *gaussian_statistics,
    float *medium_statistics,
    const void *stream);

void hybrid_backward_cuda(
    const HybridLaunchSettings &settings,
    const float *points,
    const float *densities,
    const float *medium_colors,
    const uint32_t *point_adjacency,
    const uint32_t *point_adjacency_offsets,
    const float *rays,
    const uint32_t *start_points,
    const float *hit_depths,
    const int32_t *hit_indices,
    const float *means,
    const float *log_scales,
    const float *quaternions,
    const float *gaussian_colors,
    const float *opacity_logits,
    const float *raw_kappa,
    const float *background,
    const float *output_rgba,
    const float *output_grad,
    float *points_grad,
    float *densities_grad,
    float *medium_colors_grad,
    float *means_grad,
    float *log_scales_grad,
    float *quaternions_grad,
    float *gaussian_colors_grad,
    float *opacity_logits_grad,
    float *raw_kappa_grad,
    const void *stream);

void surface_hit_cuda(uint32_t num_rays,
                      uint32_t hit_buffer_size,
                      float near_plane,
                      float far_plane,
                      const float *hit_depths,
                      const int32_t *hit_indices,
                      float *surface_depth,
                      int32_t *surface_index,
                      const void *stream);

// RGB medium: per-channel attenuation and backscatter [V,3] in one traversal.
struct RgbSceneTensors {
    const float *points;
    const float *attenuation;
    const float *backscatter;
    const float *medium_colors;
    const uint32_t *adjacency;
    const uint32_t *offsets;
    const float *rays;
    const uint32_t *start_points;
    const float *hit_depths;
    const int32_t *hit_indices;
    const float *means;
    const float *log_scales;
    const float *quaternions;
    const float *gaussian_colors;
    const float *opacity_logits;
    const float *raw_kappa;
    // Optional (nullptr = off), forward only: per-ray far plane [R], clamped
    // to settings.far_plane. Used by the visualization tracer to stop a ray
    // at an inserted object.
    const float *ray_far = nullptr;
};

struct RgbGradientTensors {
    float *points;
    float *attenuation;
    float *backscatter;
    float *medium_colors;
    float *means;
    float *log_scales;
    float *quaternions;
    float *gaussian_colors;
    float *opacity_logits;
    float *raw_kappa;
    // Optional (nullptr = off): camera right/down [2,3] and the per-Gaussian
    // sum over rays of |dL/dmean . right|, |dL/dmean . down| [N,2].
    const float *screen_axes = nullptr;
    float *means_screen_abs = nullptr;
};

void rgb_forward_cuda(
    const HybridLaunchSettings &settings,
    const RgbSceneTensors &scene,
    const float *background,
    float *direct_output,
    float *backscatter_output,
    float *transmission_output,
    uint32_t *output_num_intersections,
    const void *stream);

void rgb_backward_cuda(
    const HybridLaunchSettings &settings,
    const RgbSceneTensors &scene,
    const float *direct_output,
    const float *backscatter_output,
    const float *transmission_output,
    const float *direct_output_grad,
    const float *backscatter_output_grad,
    const float *transmission_output_grad,
    const RgbGradientTensors &gradients,
    const void *stream);

} // namespace foamsurf
