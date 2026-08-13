#include "hybrid_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace foamsurf {
namespace {

constexpr int kGaussianParameterCount = 11;
constexpr float kEpsilon = 1.0e-10f;

struct Vec3 {
    float x;
    float y;
    float z;
};

__device__ inline Vec3 make_vec3(const float *values) {
    return {values[0], values[1], values[2]};
}

__device__ inline Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ inline Vec3 operator-(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ inline Vec3 operator*(Vec3 a, float value) {
    return {a.x * value, a.y * value, a.z * value};
}

__device__ inline float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline Vec3 normalized(Vec3 value) {
    float inverse_norm = rsqrtf(fmaxf(dot(value, value), kEpsilon));
    return value * inverse_norm;
}

__device__ inline void atomic_add_vec3(float *target, Vec3 value) {
    atomicAdd(target, value.x);
    atomicAdd(target + 1, value.y);
    atomicAdd(target + 2, value.z);
}

template <int N> struct Dual {
    float value;
    float derivative[N];

    __device__ static Dual constant(float value) {
        Dual result;
        result.value = value;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            result.derivative[i] = 0.0f;
        }
        return result;
    }

    __device__ static Dual variable(float value, int index) {
        Dual result = constant(value);
        result.derivative[index] = 1.0f;
        return result;
    }
};

template <int N>
__device__ inline Dual<N> operator+(const Dual<N> &a, const Dual<N> &b) {
    Dual<N> result;
    result.value = a.value + b.value;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        result.derivative[i] = a.derivative[i] + b.derivative[i];
    }
    return result;
}

template <int N>
__device__ inline Dual<N> operator-(const Dual<N> &a, const Dual<N> &b) {
    Dual<N> result;
    result.value = a.value - b.value;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        result.derivative[i] = a.derivative[i] - b.derivative[i];
    }
    return result;
}

template <int N>
__device__ inline Dual<N> operator-(const Dual<N> &value) {
    Dual<N> result;
    result.value = -value.value;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        result.derivative[i] = -value.derivative[i];
    }
    return result;
}

template <int N>
__device__ inline Dual<N> operator*(const Dual<N> &a, const Dual<N> &b) {
    Dual<N> result;
    result.value = a.value * b.value;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        result.derivative[i] =
            a.derivative[i] * b.value + a.value * b.derivative[i];
    }
    return result;
}

template <int N>
__device__ inline Dual<N> operator/(const Dual<N> &a, const Dual<N> &b) {
    Dual<N> result;
    float inverse = 1.0f / b.value;
    result.value = a.value * inverse;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        result.derivative[i] =
            (a.derivative[i] - result.value * b.derivative[i]) * inverse;
    }
    return result;
}

template <int N>
__device__ inline Dual<N> operator+(const Dual<N> &a, float b) {
    return a + Dual<N>::constant(b);
}

template <int N>
__device__ inline Dual<N> operator-(const Dual<N> &a, float b) {
    return a - Dual<N>::constant(b);
}

template <int N>
__device__ inline Dual<N> operator*(const Dual<N> &a, float b) {
    return a * Dual<N>::constant(b);
}

template <int N>
__device__ inline Dual<N> dual_exp(const Dual<N> &input) {
    Dual<N> result;
    result.value = expf(input.value);
#pragma unroll
    for (int i = 0; i < N; ++i) {
        result.derivative[i] = result.value * input.derivative[i];
    }
    return result;
}

template <int N>
__device__ inline Dual<N> dual_log(const Dual<N> &input) {
    Dual<N> result;
    result.value = logf(input.value);
    float inverse = 1.0f / input.value;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        result.derivative[i] = input.derivative[i] * inverse;
    }
    return result;
}

template <int N>
__device__ inline Dual<N> dual_sigmoid(const Dual<N> &input) {
    Dual<N> result;
    result.value = 1.0f / (1.0f + expf(-input.value));
    float scale = result.value * (1.0f - result.value);
#pragma unroll
    for (int i = 0; i < N; ++i) {
        result.derivative[i] = scale * input.derivative[i];
    }
    return result;
}

template <int N>
__device__ inline Dual<N> dual_clamp_min(const Dual<N> &input, float minimum) {
    if (input.value >= minimum) {
        return input;
    }
    return Dual<N>::constant(minimum);
}

struct GaussianScalarEvaluation {
    float alpha;
};

__device__ inline GaussianScalarEvaluation evaluate_gaussian_scalar(
    int index,
    float hit_depth,
    Vec3 origin,
    Vec3 direction,
    const float *means,
    const float *log_scales,
    const float *quaternions,
    const float *opacity_logits,
    const float *raw_kappa) {
    Vec3 mean = make_vec3(means + 3 * index);
    const float *q = quaternions + 4 * index;
    float aa = q[0] * q[0];
    float bb = q[1] * q[1];
    float cc = q[2] * q[2];
    float dd = q[3] * q[3];
    float factor = 2.0f / fmaxf(aa + bb + cc + dd, kEpsilon);
    float bs = q[1] * factor;
    float cs = q[2] * factor;
    float ds = q[3] * factor;
    float ab = q[0] * bs;
    float ac = q[0] * cs;
    float ad = q[0] * ds;
    bb *= factor;
    float bc = q[1] * cs;
    float bd = q[1] * ds;
    cc *= factor;
    float cd = q[2] * ds;
    dd *= factor;

    Vec3 hit = origin + direction * hit_depth;
    Vec3 local = hit - mean;
    float inverse_scale_x = expf(-log_scales[2 * index]);
    float inverse_scale_y = expf(-log_scales[2 * index + 1]);
    float u = inverse_scale_x *
              ((1.0f - cc - dd) * local.x + (bc + ad) * local.y +
               (bd - ac) * local.z);
    float v = inverse_scale_y *
              ((bc - ad) * local.x + (1.0f - bb - dd) * local.y +
               (cd + ab) * local.z);
    float radius_squared = u * u + v * v;
    float kappa = 1.0f + log1pf(expf(raw_kappa[index]));
    float radial_power = powf(fmaxf(radius_squared, kEpsilon), kappa);
    float response = expf(-0.5f * radial_power / kappa);
    float opacity = 1.0f / (1.0f + expf(-opacity_logits[index]));
    return {opacity * response};
}

struct GaussianDualEvaluation {
    Dual<kGaussianParameterCount> depth;
    Dual<kGaussianParameterCount> alpha;
};

__device__ inline GaussianDualEvaluation evaluate_gaussian_dual(
    int index,
    float saved_hit_depth,
    Vec3 origin,
    Vec3 direction,
    const float *means,
    const float *log_scales,
    const float *quaternions,
    const float *opacity_logits,
    const float *raw_kappa) {
    using D = Dual<kGaussianParameterCount>;
    D mean[3] = {
        D::variable(means[3 * index], 0),
        D::variable(means[3 * index + 1], 1),
        D::variable(means[3 * index + 2], 2),
    };
    D scales[2] = {
        D::variable(log_scales[2 * index], 3),
        D::variable(log_scales[2 * index + 1], 4),
    };
    D q[4] = {
        D::variable(quaternions[4 * index], 5),
        D::variable(quaternions[4 * index + 1], 6),
        D::variable(quaternions[4 * index + 2], 7),
        D::variable(quaternions[4 * index + 3], 8),
    };
    D opacity_logit = D::variable(opacity_logits[index], 9);
    D raw_shape = D::variable(raw_kappa[index], 10);

    D aa = q[0] * q[0];
    D bb = q[1] * q[1];
    D cc = q[2] * q[2];
    D dd = q[3] * q[3];
    D factor = D::constant(2.0f) / (aa + bb + cc + dd);
    D bs = q[1] * factor;
    D cs = q[2] * factor;
    D ds = q[3] * factor;
    D ab = q[0] * bs;
    D ac = q[0] * cs;
    D ad = q[0] * ds;
    bb = bb * factor;
    D bc = q[1] * cs;
    D bd = q[1] * ds;
    cc = cc * factor;
    D cd = q[2] * ds;
    dd = dd * factor;

    D row0[3] = {D::constant(1.0f) - cc - dd, bc + ad, bd - ac};
    D row1[3] = {bc - ad, D::constant(1.0f) - bb - dd, cd + ab};
    D row2[3] = {bd + ac, cd - ab, D::constant(1.0f) - bb - cc};
    D numerator = row2[0] * (mean[0] - origin.x) +
                  row2[1] * (mean[1] - origin.y) +
                  row2[2] * (mean[2] - origin.z);
    D denominator = row2[0] * direction.x + row2[1] * direction.y +
                    row2[2] * direction.z;
    D depth = numerator / denominator;
    depth.value = saved_hit_depth;

    D local[3] = {
        D::constant(origin.x) + depth * direction.x - mean[0],
        D::constant(origin.y) + depth * direction.y - mean[1],
        D::constant(origin.z) + depth * direction.z - mean[2],
    };
    D u = dual_exp(-scales[0]) *
          (row0[0] * local[0] + row0[1] * local[1] + row0[2] * local[2]);
    D v = dual_exp(-scales[1]) *
          (row1[0] * local[0] + row1[1] * local[1] + row1[2] * local[2]);
    D radius_squared = dual_clamp_min(u * u + v * v, kEpsilon);
    D kappa = D::constant(1.0f) + dual_log(D::constant(1.0f) + dual_exp(raw_shape));
    D radial_power = dual_exp(kappa * dual_log(radius_squared));
    D response = dual_exp(D::constant(-0.5f) * radial_power / kappa);
    D alpha = dual_sigmoid(opacity_logit) * response;
    return {depth, alpha};
}

struct HitCursor {
    uint32_t slot;
};

__device__ inline bool next_hit(
    const HybridLaunchSettings &settings,
    uint32_t ray_index,
    HitCursor &cursor,
    const float *hit_depths,
    const int32_t *hit_indices,
    float minimum_depth,
    float maximum_depth,
    float &depth,
    int &index) {
    while (cursor.slot < settings.hit_buffer_size) {
        uint32_t offset = cursor.slot * settings.num_rays + ray_index;
        int candidate_index = hit_indices[offset];
        float candidate_depth = hit_depths[offset];
        if (candidate_index < 0 || candidate_depth < minimum_depth - 1.0e-6f ||
            candidate_depth < settings.near_plane) {
            ++cursor.slot;
            continue;
        }
        if (candidate_depth > maximum_depth ||
            candidate_depth > settings.far_plane) {
            return false;
        }
        depth = candidate_depth;
        index = candidate_index;
        return true;
    }
    return false;
}

__device__ inline bool next_boundary(
    uint32_t current,
    Vec3 origin,
    Vec3 direction,
    const float *points,
    const uint32_t *adjacency,
    const uint32_t *offsets,
    float &depth,
    uint32_t &next) {
    Vec3 point = make_vec3(points + 3 * current);
    uint32_t begin = offsets[current];
    uint32_t end = offsets[current + 1];
    depth = INFINITY;
    next = UINT32_MAX;
    for (uint32_t edge = begin; edge < end; ++edge) {
        uint32_t neighbor = adjacency[edge];
        Vec3 opposite = make_vec3(points + 3 * neighbor);
        Vec3 difference = opposite - point;
        float denominator = dot(difference, direction);
        if (denominator <= 0.0f) {
            continue;
        }
        Vec3 face_origin = point + difference * 0.5f;
        float candidate = dot(face_origin - origin, difference) / denominator;
        if (candidate < depth) {
            depth = candidate;
            next = neighbor;
        }
    }
    return next != UINT32_MAX;
}

__device__ inline void composite_event(
    float alpha, Vec3 color, Vec3 &rgb, float &transmittance) {
    alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
    float weight = transmittance * alpha;
    rgb = rgb + color * weight;
    transmittance *= 1.0f - alpha;
}

__device__ inline void composite_medium(
    uint32_t cell,
    float start,
    float end,
    const float *densities,
    const float *colors,
    Vec3 &rgb,
    float &transmittance) {
    float length = fmaxf(end - start, 0.0f);
    if (length <= 0.0f) {
        return;
    }
    float density = fmaxf(densities[cell], 0.0f);
    float alpha = 1.0f - expf(-density * length);
    composite_event(alpha, make_vec3(colors + 3 * cell), rgb, transmittance);
}

__global__ void hybrid_forward_kernel(
    HybridLaunchSettings settings,
    const float *points,
    const float *densities,
    const float *medium_colors,
    const uint32_t *adjacency,
    const uint32_t *offsets,
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
    float *output,
    uint32_t *num_intersections) {
    uint32_t ray_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_index >= settings.num_rays) {
        return;
    }
    Vec3 origin = make_vec3(rays + 6 * ray_index);
    Vec3 direction = normalized(make_vec3(rays + 6 * ray_index + 3));
    Vec3 rgb = {0.0f, 0.0f, 0.0f};
    float transmittance = 1.0f;
    float cursor_depth = settings.near_plane;
    HitCursor hit_cursor{0};
    uint32_t steps = 0;

    uint32_t current = settings.num_points ? start_points[ray_index] : 0;
    bool finished = false;
    while (!finished && cursor_depth < settings.far_plane &&
           transmittance > settings.transmittance_threshold &&
           steps < settings.max_intersections) {
        ++steps;
        float boundary_depth = settings.far_plane;
        uint32_t next = UINT32_MAX;
        bool has_boundary = false;
        if (settings.num_points) {
            has_boundary = next_boundary(current,
                                         origin,
                                         direction,
                                         points,
                                         adjacency,
                                         offsets,
                                         boundary_depth,
                                         next);
            if (!has_boundary || boundary_depth >= settings.far_plane) {
                boundary_depth = settings.far_plane;
                finished = true;
            } else if (boundary_depth <= cursor_depth) {
                current = next;
                continue;
            }
        } else {
            finished = true;
        }

        float hit_depth;
        int gaussian_index;
        while (next_hit(settings,
                        ray_index,
                        hit_cursor,
                        hit_depths,
                        hit_indices,
                        cursor_depth,
                        boundary_depth,
                        hit_depth,
                        gaussian_index)) {
            if (settings.num_points) {
                composite_medium(current,
                                 cursor_depth,
                                 hit_depth,
                                 densities,
                                 medium_colors,
                                 rgb,
                                 transmittance);
            }
            auto evaluation = evaluate_gaussian_scalar(gaussian_index,
                                                       hit_depth,
                                                       origin,
                                                       direction,
                                                       means,
                                                       log_scales,
                                                       quaternions,
                                                       opacity_logits,
                                                       raw_kappa);
            composite_event(evaluation.alpha,
                            make_vec3(gaussian_colors + 3 * gaussian_index),
                            rgb,
                            transmittance);
            cursor_depth = fmaxf(cursor_depth, hit_depth);
            ++hit_cursor.slot;
            if (transmittance <= settings.transmittance_threshold) {
                break;
            }
        }
        if (transmittance <= settings.transmittance_threshold) {
            break;
        }
        if (settings.num_points) {
            composite_medium(current,
                             cursor_depth,
                             boundary_depth,
                             densities,
                             medium_colors,
                             rgb,
                             transmittance);
        }
        cursor_depth = boundary_depth;
        if (!finished && has_boundary) {
            current = next;
        }
    }

    Vec3 background_color = make_vec3(background);
    rgb = rgb + background_color * transmittance;
    output[4 * ray_index] = rgb.x;
    output[4 * ray_index + 1] = rgb.y;
    output[4 * ray_index + 2] = rgb.z;
    output[4 * ray_index + 3] = 1.0f - transmittance;
    num_intersections[ray_index] = steps;
}

// Optional B0.1 instrumentation. It repeats scalar traversal in a separate
// forward-only launch, so regular rendering and autograd remain unchanged.
// Outputs are reduced to [R,8], [G,4], [V,5], never rays x primitives.
__global__ void hybrid_diagnostics_kernel(
    HybridLaunchSettings settings,
    const float *points,
    const float *densities,
    const uint32_t *adjacency,
    const uint32_t *offsets,
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
    float *medium_statistics) {
    uint32_t ray_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_index >= settings.num_rays) {
        return;
    }
    Vec3 origin = make_vec3(rays + 6 * ray_index);
    Vec3 direction = normalized(make_vec3(rays + 6 * ray_index + 3));
    float transmittance = 1.0f;
    float cursor_depth = settings.near_plane;
    HitCursor hit_cursor{0};
    uint32_t steps = 0;
    uint32_t gaussian_events = 0;
    uint32_t medium_segments = 0;
    float gaussian_weight = 0.0f;
    float medium_weight = 0.0f;
    float medium_path = 0.0f;
    float weighted_medium_path = 0.0f;

    uint32_t current = settings.num_points ? start_points[ray_index] : 0;
    bool finished = false;
    while (!finished && cursor_depth < settings.far_plane &&
           transmittance > settings.transmittance_threshold &&
           steps < settings.max_intersections) {
        ++steps;
        float boundary_depth = settings.far_plane;
        uint32_t next = UINT32_MAX;
        bool has_boundary = false;
        if (settings.num_points) {
            has_boundary = next_boundary(current, origin, direction, points,
                                         adjacency, offsets, boundary_depth, next);
            if (!has_boundary || boundary_depth >= settings.far_plane) {
                boundary_depth = settings.far_plane;
                finished = true;
            } else if (boundary_depth <= cursor_depth) {
                current = next;
                continue;
            }
        } else {
            finished = true;
        }

        float hit_depth;
        int gaussian_index;
        while (next_hit(settings, ray_index, hit_cursor, hit_depths, hit_indices,
                        cursor_depth, boundary_depth, hit_depth,
                        gaussian_index)) {
            if (settings.num_points) {
                float length = fmaxf(hit_depth - cursor_depth, 0.0f);
                if (length > 0.0f) {
                    float transmittance_before = transmittance;
                    float density = fmaxf(densities[current], 0.0f);
                    float alpha = 1.0f - expf(-density * length);
                    float weight = transmittance_before * alpha;
                    medium_weight += weight;
                    medium_path += length;
                    weighted_medium_path += transmittance_before * length;
                    ++medium_segments;
                    atomicAdd(medium_statistics + 5 * current, weight);
                    atomicAdd(medium_statistics + 5 * current + 1, 1.0f);
                    atomicAdd(medium_statistics + 5 * current + 2, length);
                    atomicAdd(medium_statistics + 5 * current + 3,
                              transmittance_before * length);
                    atomicMax(
                        reinterpret_cast<int *>(
                            medium_statistics + 5 * current + 4),
                        __float_as_int(ray_errors[ray_index]));
                    transmittance *= 1.0f - alpha;
                }
            }
            auto evaluation = evaluate_gaussian_scalar(
                gaussian_index, hit_depth, origin, direction, means, log_scales,
                quaternions, opacity_logits, raw_kappa);
            float alpha = fminf(fmaxf(evaluation.alpha, 0.0f), 1.0f);
            float weight = transmittance * alpha;
            gaussian_weight += weight;
            ++gaussian_events;
            atomicAdd(gaussian_statistics + 4 * gaussian_index, weight);
            atomicAdd(gaussian_statistics + 4 * gaussian_index + 1, 1.0f);
            atomicAdd(gaussian_statistics + 4 * gaussian_index + 2, alpha);
            atomicMax(
                reinterpret_cast<int *>(
                    gaussian_statistics + 4 * gaussian_index + 3),
                __float_as_int(ray_errors[ray_index]));
            transmittance *= 1.0f - alpha;
            cursor_depth = fmaxf(cursor_depth, hit_depth);
            ++hit_cursor.slot;
            if (transmittance <= settings.transmittance_threshold) {
                break;
            }
        }
        if (transmittance <= settings.transmittance_threshold) {
            break;
        }
        if (settings.num_points) {
            float length = fmaxf(boundary_depth - cursor_depth, 0.0f);
            if (length > 0.0f) {
                float transmittance_before = transmittance;
                float density = fmaxf(densities[current], 0.0f);
                float alpha = 1.0f - expf(-density * length);
                float weight = transmittance_before * alpha;
                medium_weight += weight;
                medium_path += length;
                weighted_medium_path += transmittance_before * length;
                ++medium_segments;
                atomicAdd(medium_statistics + 5 * current, weight);
                atomicAdd(medium_statistics + 5 * current + 1, 1.0f);
                atomicAdd(medium_statistics + 5 * current + 2, length);
                atomicAdd(medium_statistics + 5 * current + 3,
                          transmittance_before * length);
                atomicMax(
                    reinterpret_cast<int *>(
                        medium_statistics + 5 * current + 4),
                        __float_as_int(ray_errors[ray_index]));
                transmittance *= 1.0f - alpha;
            }
        }
        cursor_depth = boundary_depth;
        if (!finished && has_boundary) {
            current = next;
        }
    }

    float *stats = ray_statistics + 8 * ray_index;
    stats[0] = gaussian_weight;
    stats[1] = medium_weight;
    stats[2] = medium_path;
    stats[3] = weighted_medium_path;
    stats[4] = static_cast<float>(gaussian_events);
    stats[5] = static_cast<float>(medium_segments);
    stats[6] = transmittance;
    stats[7] = (!finished && cursor_depth < settings.far_plane &&
                transmittance > settings.transmittance_threshold &&
                steps >= settings.max_intersections) ? 1.0f : 0.0f;
}

enum class OwnerType : int { None = 0, Gaussian = 1, Boundary = 2 };

struct EndpointOwner {
    OwnerType type;
    int gaussian;
    uint32_t first;
    uint32_t second;
    float depth;
};

__device__ inline Vec3 cell_intersection_gradient(
    Vec3 primal, Vec3 opposite, Vec3 origin, Vec3 direction) {
    Vec3 face_origin = (primal + opposite) * 0.5f;
    Vec3 normal = opposite - primal;
    float numerator = dot(face_origin - origin, normal);
    float denominator = dot(normal, direction);
    Vec3 result = direction * numerator + (origin - primal) * denominator;
    return result * (1.0f / (denominator * denominator));
}

__device__ inline void accumulate_gaussian_geometry(
    int index,
    float hit_depth,
    float alpha_coefficient,
    float depth_coefficient,
    Vec3 origin,
    Vec3 direction,
    const float *means,
    const float *log_scales,
    const float *quaternions,
    const float *opacity_logits,
    const float *raw_kappa,
    float *means_grad,
    float *log_scales_grad,
    float *quaternions_grad,
    float *opacity_logits_grad,
    float *raw_kappa_grad) {
    auto evaluation = evaluate_gaussian_dual(index,
                                             hit_depth,
                                             origin,
                                             direction,
                                             means,
                                             log_scales,
                                             quaternions,
                                             opacity_logits,
                                             raw_kappa);
    float gradient[kGaussianParameterCount];
#pragma unroll
    for (int i = 0; i < kGaussianParameterCount; ++i) {
        gradient[i] = alpha_coefficient * evaluation.alpha.derivative[i] +
                      depth_coefficient * evaluation.depth.derivative[i];
    }
    for (int i = 0; i < 3; ++i) {
        atomicAdd(means_grad + 3 * index + i, gradient[i]);
    }
    for (int i = 0; i < 2; ++i) {
        atomicAdd(log_scales_grad + 2 * index + i, gradient[3 + i]);
    }
    for (int i = 0; i < 4; ++i) {
        atomicAdd(quaternions_grad + 4 * index + i, gradient[5 + i]);
    }
    atomicAdd(opacity_logits_grad + index, gradient[9]);
    atomicAdd(raw_kappa_grad + index, gradient[10]);
}

__device__ inline void accumulate_endpoint_gradient(
    const EndpointOwner &owner,
    float coefficient,
    Vec3 origin,
    Vec3 direction,
    const float *points,
    const float *means,
    const float *log_scales,
    const float *quaternions,
    const float *opacity_logits,
    const float *raw_kappa,
    float *points_grad,
    float *means_grad,
    float *log_scales_grad,
    float *quaternions_grad,
    float *opacity_logits_grad,
    float *raw_kappa_grad) {
    if (coefficient == 0.0f || owner.type == OwnerType::None) {
        return;
    }
    if (owner.type == OwnerType::Gaussian) {
        accumulate_gaussian_geometry(owner.gaussian,
                                     owner.depth,
                                     0.0f,
                                     coefficient,
                                     origin,
                                     direction,
                                     means,
                                     log_scales,
                                     quaternions,
                                     opacity_logits,
                                     raw_kappa,
                                     means_grad,
                                     log_scales_grad,
                                     quaternions_grad,
                                     opacity_logits_grad,
                                     raw_kappa_grad);
        return;
    }
    Vec3 first = make_vec3(points + 3 * owner.first);
    Vec3 second = make_vec3(points + 3 * owner.second);
    Vec3 first_gradient =
        cell_intersection_gradient(first, second, origin, direction);
    Vec3 second_gradient =
        cell_intersection_gradient(second, first, origin, direction);
    atomic_add_vec3(points_grad + 3 * owner.first,
                    first_gradient * coefficient);
    atomic_add_vec3(points_grad + 3 * owner.second,
                    second_gradient * coefficient);
}

struct BackwardState {
    Vec3 prefix;
    float transmittance;
};

__device__ inline float backward_event(
    float alpha,
    Vec3 color,
    Vec3 final_rgb,
    float final_transmittance,
    Vec3 rgb_gradient,
    float alpha_gradient,
    BackwardState &state,
    Vec3 &color_gradient) {
    alpha = fminf(fmaxf(alpha, 0.0f), 1.0f - 1.0e-7f);
    float transmittance_before = state.transmittance;
    float weight = transmittance_before * alpha;
    color_gradient = rgb_gradient * weight;
    Vec3 prefix_after = state.prefix + color * weight;
    float transmittance_after = transmittance_before * (1.0f - alpha);
    Vec3 tail = (final_rgb - prefix_after) *
                (1.0f / fmaxf(transmittance_after, kEpsilon));
    float alpha_derivative =
        transmittance_before * dot(rgb_gradient, color - tail) +
        alpha_gradient * final_transmittance /
            fmaxf(1.0f - alpha, kEpsilon);
    state.prefix = prefix_after;
    state.transmittance = transmittance_after;
    return alpha_derivative;
}

__device__ inline void backward_medium(
    uint32_t cell,
    float start,
    float end,
    const EndpointOwner &start_owner,
    const EndpointOwner &end_owner,
    Vec3 origin,
    Vec3 direction,
    Vec3 final_rgb,
    float final_transmittance,
    Vec3 rgb_gradient,
    float alpha_gradient,
    const float *points,
    const float *densities,
    const float *medium_colors,
    const float *means,
    const float *log_scales,
    const float *quaternions,
    const float *opacity_logits,
    const float *raw_kappa,
    float *points_grad,
    float *densities_grad,
    float *medium_colors_grad,
    float *means_grad,
    float *log_scales_grad,
    float *quaternions_grad,
    float *opacity_logits_grad,
    float *raw_kappa_grad,
    BackwardState &state) {
    float length = fmaxf(end - start, 0.0f);
    if (length <= 0.0f) {
        return;
    }
    float raw_density = densities[cell];
    float density = fmaxf(raw_density, 0.0f);
    float one_minus_alpha = expf(-density * length);
    float alpha = 1.0f - one_minus_alpha;
    Vec3 local_color_gradient;
    float alpha_derivative = backward_event(alpha,
                                             make_vec3(medium_colors + 3 * cell),
                                             final_rgb,
                                             final_transmittance,
                                             rgb_gradient,
                                             alpha_gradient,
                                             state,
                                             local_color_gradient);
    atomic_add_vec3(medium_colors_grad + 3 * cell, local_color_gradient);
    if (raw_density > 0.0f) {
        atomicAdd(densities_grad + cell,
                  alpha_derivative * length * one_minus_alpha);
    }
    float length_derivative = alpha_derivative * density * one_minus_alpha;
    accumulate_endpoint_gradient(start_owner,
                                 -length_derivative,
                                 origin,
                                 direction,
                                 points,
                                 means,
                                 log_scales,
                                 quaternions,
                                 opacity_logits,
                                 raw_kappa,
                                 points_grad,
                                 means_grad,
                                 log_scales_grad,
                                 quaternions_grad,
                                 opacity_logits_grad,
                                 raw_kappa_grad);
    accumulate_endpoint_gradient(end_owner,
                                 length_derivative,
                                 origin,
                                 direction,
                                 points,
                                 means,
                                 log_scales,
                                 quaternions,
                                 opacity_logits,
                                 raw_kappa,
                                 points_grad,
                                 means_grad,
                                 log_scales_grad,
                                 quaternions_grad,
                                 opacity_logits_grad,
                                 raw_kappa_grad);
}

__global__ void hybrid_backward_kernel(
    HybridLaunchSettings settings,
    const float *points,
    const float *densities,
    const float *medium_colors,
    const uint32_t *adjacency,
    const uint32_t *offsets,
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
    const float *output,
    const float *output_grad,
    float *points_grad,
    float *densities_grad,
    float *medium_colors_grad,
    float *means_grad,
    float *log_scales_grad,
    float *quaternions_grad,
    float *gaussian_colors_grad,
    float *opacity_logits_grad,
    float *raw_kappa_grad) {
    uint32_t ray_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_index >= settings.num_rays) {
        return;
    }
    Vec3 origin = make_vec3(rays + 6 * ray_index);
    Vec3 direction = normalized(make_vec3(rays + 6 * ray_index + 3));
    Vec3 final_rgb = make_vec3(output + 4 * ray_index);
    float final_transmittance = 1.0f - output[4 * ray_index + 3];
    Vec3 rgb_gradient = make_vec3(output_grad + 4 * ray_index);
    float alpha_gradient = output_grad[4 * ray_index + 3];
    BackwardState state{{0.0f, 0.0f, 0.0f}, 1.0f};
    float cursor_depth = settings.near_plane;
    EndpointOwner cursor_owner{OwnerType::None, -1, 0, 0, cursor_depth};
    HitCursor hit_cursor{0};
    uint32_t steps = 0;
    uint32_t current = settings.num_points ? start_points[ray_index] : 0;
    bool finished = false;

    while (!finished && cursor_depth < settings.far_plane &&
           state.transmittance > settings.transmittance_threshold &&
           steps < settings.max_intersections) {
        ++steps;
        float boundary_depth = settings.far_plane;
        uint32_t next = UINT32_MAX;
        bool has_boundary = false;
        EndpointOwner boundary_owner{
            OwnerType::None, -1, 0, 0, settings.far_plane};
        if (settings.num_points) {
            has_boundary = next_boundary(current,
                                         origin,
                                         direction,
                                         points,
                                         adjacency,
                                         offsets,
                                         boundary_depth,
                                         next);
            if (!has_boundary || boundary_depth >= settings.far_plane) {
                boundary_depth = settings.far_plane;
                finished = true;
            } else if (boundary_depth <= cursor_depth) {
                current = next;
                continue;
            } else {
                boundary_owner = {OwnerType::Boundary,
                                  -1,
                                  current,
                                  next,
                                  boundary_depth};
            }
        } else {
            finished = true;
        }

        float hit_depth;
        int gaussian_index;
        while (next_hit(settings,
                        ray_index,
                        hit_cursor,
                        hit_depths,
                        hit_indices,
                        cursor_depth,
                        boundary_depth,
                        hit_depth,
                        gaussian_index)) {
            EndpointOwner gaussian_owner{
                OwnerType::Gaussian, gaussian_index, 0, 0, hit_depth};
            if (settings.num_points) {
                backward_medium(current,
                                cursor_depth,
                                hit_depth,
                                cursor_owner,
                                gaussian_owner,
                                origin,
                                direction,
                                final_rgb,
                                final_transmittance,
                                rgb_gradient,
                                alpha_gradient,
                                points,
                                densities,
                                medium_colors,
                                means,
                                log_scales,
                                quaternions,
                                opacity_logits,
                                raw_kappa,
                                points_grad,
                                densities_grad,
                                medium_colors_grad,
                                means_grad,
                                log_scales_grad,
                                quaternions_grad,
                                opacity_logits_grad,
                                raw_kappa_grad,
                                state);
            }
            auto evaluation = evaluate_gaussian_scalar(gaussian_index,
                                                       hit_depth,
                                                       origin,
                                                       direction,
                                                       means,
                                                       log_scales,
                                                       quaternions,
                                                       opacity_logits,
                                                       raw_kappa);
            Vec3 local_color_gradient;
            float alpha_derivative = backward_event(
                evaluation.alpha,
                make_vec3(gaussian_colors + 3 * gaussian_index),
                final_rgb,
                final_transmittance,
                rgb_gradient,
                alpha_gradient,
                state,
                local_color_gradient);
            atomic_add_vec3(gaussian_colors_grad + 3 * gaussian_index,
                            local_color_gradient);
            accumulate_gaussian_geometry(gaussian_index,
                                         hit_depth,
                                         alpha_derivative,
                                         0.0f,
                                         origin,
                                         direction,
                                         means,
                                         log_scales,
                                         quaternions,
                                         opacity_logits,
                                         raw_kappa,
                                         means_grad,
                                         log_scales_grad,
                                         quaternions_grad,
                                         opacity_logits_grad,
                                         raw_kappa_grad);
            cursor_depth = fmaxf(cursor_depth, hit_depth);
            cursor_owner = gaussian_owner;
            ++hit_cursor.slot;
            if (state.transmittance <= settings.transmittance_threshold) {
                break;
            }
        }
        if (state.transmittance <= settings.transmittance_threshold) {
            break;
        }
        if (settings.num_points) {
            backward_medium(current,
                            cursor_depth,
                            boundary_depth,
                            cursor_owner,
                            boundary_owner,
                            origin,
                            direction,
                            final_rgb,
                            final_transmittance,
                            rgb_gradient,
                            alpha_gradient,
                            points,
                            densities,
                            medium_colors,
                            means,
                            log_scales,
                            quaternions,
                            opacity_logits,
                            raw_kappa,
                            points_grad,
                            densities_grad,
                            medium_colors_grad,
                            means_grad,
                            log_scales_grad,
                            quaternions_grad,
                            opacity_logits_grad,
                            raw_kappa_grad,
                            state);
        }
        cursor_depth = boundary_depth;
        cursor_owner = boundary_owner;
        if (!finished && has_boundary) {
            current = next;
        }
    }
}

} // namespace

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
    const void *stream) {
    constexpr uint32_t block_size = 128;
    uint32_t grid_size = (settings.num_rays + block_size - 1) / block_size;
    cudaStream_t cuda_stream =
        *reinterpret_cast<const cudaStream_t *>(stream);
    hybrid_forward_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        settings,
        points,
        densities,
        medium_colors,
        point_adjacency,
        point_adjacency_offsets,
        rays,
        start_points,
        hit_depths,
        hit_indices,
        means,
        log_scales,
        quaternions,
        gaussian_colors,
        opacity_logits,
        raw_kappa,
        background,
        output_rgba,
        output_num_intersections);
}

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
    const void *stream) {
    constexpr uint32_t block_size = 128;
    uint32_t grid_size = (settings.num_rays + block_size - 1) / block_size;
    cudaStream_t cuda_stream =
        *reinterpret_cast<const cudaStream_t *>(stream);
    hybrid_diagnostics_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        settings, points, densities, point_adjacency, point_adjacency_offsets,
        rays, start_points, hit_depths, hit_indices, means, log_scales,
        quaternions, opacity_logits, raw_kappa, ray_errors, ray_statistics,
        gaussian_statistics, medium_statistics);
}

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
    const float *,
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
    const void *stream) {
    constexpr uint32_t block_size = 128;
    uint32_t grid_size = (settings.num_rays + block_size - 1) / block_size;
    cudaStream_t cuda_stream =
        *reinterpret_cast<const cudaStream_t *>(stream);
    hybrid_backward_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        settings,
        points,
        densities,
        medium_colors,
        point_adjacency,
        point_adjacency_offsets,
        rays,
        start_points,
        hit_depths,
        hit_indices,
        means,
        log_scales,
        quaternions,
        gaussian_colors,
        opacity_logits,
        raw_kappa,
        output_rgba,
        output_grad,
        points_grad,
        densities_grad,
        medium_colors_grad,
        means_grad,
        log_scales_grad,
        quaternions_grad,
        gaussian_colors_grad,
        opacity_logits_grad,
        raw_kappa_grad);
}

} // namespace foamsurf
