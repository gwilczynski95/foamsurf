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

} // namespace foamsurf
