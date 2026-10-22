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

// Medium statistics of the cell a ray is in, summed while it stays there and
// written when it leaves (one atomicMax per visit: the ray error is constant).
struct CellStatistics {
    uint32_t cell;
    float weight;
    float segments;
    float path;
    float weighted_path;
};

__device__ inline void flush_cell_statistics(CellStatistics &statistics,
                                             float ray_error,
                                             float *medium_statistics) {
    if (statistics.cell == UINT32_MAX) {
        return;
    }
    float *target = medium_statistics + 5 * statistics.cell;
    atomicAdd(target, statistics.weight);
    atomicAdd(target + 1, statistics.segments);
    atomicAdd(target + 2, statistics.path);
    atomicAdd(target + 3, statistics.weighted_path);
    atomicMax(reinterpret_cast<int *>(target + 4), __float_as_int(ray_error));
    statistics = {UINT32_MAX, 0.0f, 0.0f, 0.0f, 0.0f};
}

__device__ inline void add_cell_statistics(CellStatistics &statistics,
                                           uint32_t cell,
                                           float weight,
                                           float length,
                                           float transmittance_before,
                                           float ray_error,
                                           float *medium_statistics) {
    if (statistics.cell != cell) {
        flush_cell_statistics(statistics, ray_error, medium_statistics);
        statistics.cell = cell;
    }
    statistics.weight += weight;
    statistics.segments += 1.0f;
    statistics.path += length;
    statistics.weighted_path += transmittance_before * length;
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
    float ray_error = ray_errors[ray_index];
    CellStatistics cell_statistics{UINT32_MAX, 0.0f, 0.0f, 0.0f, 0.0f};

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
                    add_cell_statistics(cell_statistics, current, weight,
                                        length, transmittance_before,
                                        ray_error, medium_statistics);
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
                add_cell_statistics(cell_statistics, current, weight, length,
                                    transmittance_before, ray_error,
                                    medium_statistics);
                transmittance *= 1.0f - alpha;
            }
        }
        cursor_depth = boundary_depth;
        if (!finished && has_boundary) {
            current = next;
        }
    }

    flush_cell_statistics(cell_statistics, ray_error, medium_statistics);
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

// First valid Gaussian hit per ray: the smallest depth among slots with an
// index >= 0 and near <= depth <= far, the earliest slot on ties, slot 0 when
// no slot is valid (then depth -1), exactly as the former PyTorch where/min/
// gather over the [K, R] buffers.
__global__ void surface_hit_kernel(uint32_t num_rays,
                                   uint32_t hit_buffer_size,
                                   float near_plane,
                                   float far_plane,
                                   const float *hit_depths,
                                   const int32_t *hit_indices,
                                   float *surface_depth,
                                   int32_t *surface_index) {
    uint32_t ray_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_index >= num_rays) {
        return;
    }
    float best_depth = INFINITY;
    uint32_t best_slot = 0;
    for (uint32_t slot = 0; slot < hit_buffer_size; ++slot) {
        uint32_t offset = slot * num_rays + ray_index;
        float depth = hit_depths[offset];
        bool valid = hit_indices[offset] >= 0 && depth >= near_plane &&
                     depth <= far_plane;
        if (valid && depth < best_depth) {
            best_depth = depth;
            best_slot = slot;
        }
    }
    surface_index[ray_index] = hit_indices[best_slot * num_rays + ray_index];
    surface_depth[ray_index] = isfinite(best_depth) ? best_depth : -1.0f;
}

// RGB medium in one traversal. It computes what rgb_medium.compose_cuda gets
// from six launches of the scalar kernel above. Pass p < 3 is the direct pass
// of channel p (density attenuation[:, p], medium colour 0); pass p >= 3 is the
// backscatter pass of channel p - 3 (density backscatter[:, p - 3], Gaussian
// colour and background 0). Every pass keeps its own transmittance, updated
// with the same float operations as the scalar kernel, and stops at the same
// check points (after a Gaussian event, before a Voronoi step), so each output
// equals the scalar launch's output for its channel. Boundaries and Gaussian
// alphas are computed once for all passes.
constexpr int kRgbPasses = 6;
constexpr uint32_t kAllRgbPasses = (1u << kRgbPasses) - 1u;

__device__ inline float component(Vec3 value, int index) {
    return index == 0 ? value.x : (index == 1 ? value.y : value.z);
}

__device__ inline float rgb_pass_density(const float *attenuation,
                                         const float *backscatter,
                                         uint32_t cell,
                                         int pass) {
    return pass < 3 ? attenuation[3 * cell + pass]
                    : backscatter[3 * cell + pass - 3];
}

__device__ inline uint32_t rgb_active_passes(const float *transmittance,
                                             uint32_t active,
                                             float threshold) {
#pragma unroll
    for (int pass = 0; pass < kRgbPasses; ++pass) {
        if (transmittance[pass] <= threshold) {
            active &= ~(1u << pass);
        }
    }
    return active;
}

// composite_medium for every active pass; only backscatter passes have colour.
__device__ inline void rgb_composite_medium(uint32_t cell,
                                            float start,
                                            float end,
                                            const float *attenuation,
                                            const float *backscatter,
                                            const float *medium_colors,
                                            uint32_t active,
                                            float *value,
                                            float *transmittance) {
    float length = fmaxf(end - start, 0.0f);
    if (length <= 0.0f) {
        return;
    }
    Vec3 color = make_vec3(medium_colors + 3 * cell);
#pragma unroll
    for (int pass = 0; pass < kRgbPasses; ++pass) {
        if (!(active & (1u << pass))) {
            continue;
        }
        float density = fmaxf(
            rgb_pass_density(attenuation, backscatter, cell, pass), 0.0f);
        float alpha = 1.0f - expf(-density * length);
        alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
        float weight = transmittance[pass] * alpha;
        if (pass >= 3) {
            value[pass] = value[pass] + component(color, pass - 3) * weight;
        }
        transmittance[pass] *= 1.0f - alpha;
    }
}

__global__ void rgb_forward_kernel(
    HybridLaunchSettings settings,
    const float *points,
    const float *attenuation,
    const float *backscatter,
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
    float *direct_output,
    float *backscatter_output,
    float *transmission_output,
    uint32_t *num_intersections) {
    uint32_t ray_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_index >= settings.num_rays) {
        return;
    }
    Vec3 origin = make_vec3(rays + 6 * ray_index);
    Vec3 direction = normalized(make_vec3(rays + 6 * ray_index + 3));
    float value[kRgbPasses];
    float transmittance[kRgbPasses];
#pragma unroll
    for (int pass = 0; pass < kRgbPasses; ++pass) {
        value[pass] = 0.0f;
        transmittance[pass] = 1.0f;
    }
    uint32_t active = rgb_active_passes(
        transmittance, kAllRgbPasses, settings.transmittance_threshold);
    float cursor_depth = settings.near_plane;
    HitCursor hit_cursor{0};
    uint32_t steps = 0;

    uint32_t current = settings.num_points ? start_points[ray_index] : 0;
    bool finished = false;
    while (!finished && cursor_depth < settings.far_plane && active &&
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
                rgb_composite_medium(current,
                                     cursor_depth,
                                     hit_depth,
                                     attenuation,
                                     backscatter,
                                     medium_colors,
                                     active,
                                     value,
                                     transmittance);
            }
            float alpha = evaluate_gaussian_scalar(gaussian_index,
                                                   hit_depth,
                                                   origin,
                                                   direction,
                                                   means,
                                                   log_scales,
                                                   quaternions,
                                                   opacity_logits,
                                                   raw_kappa)
                              .alpha;
            alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
            Vec3 color = make_vec3(gaussian_colors + 3 * gaussian_index);
#pragma unroll
            for (int pass = 0; pass < kRgbPasses; ++pass) {
                if (!(active & (1u << pass))) {
                    continue;
                }
                float weight = transmittance[pass] * alpha;
                if (pass < 3) {
                    value[pass] = value[pass] + component(color, pass) * weight;
                }
                transmittance[pass] *= 1.0f - alpha;
            }
            active = rgb_active_passes(
                transmittance, active, settings.transmittance_threshold);
            cursor_depth = fmaxf(cursor_depth, hit_depth);
            ++hit_cursor.slot;
            if (!active) {
                break;
            }
        }
        if (!active) {
            break;
        }
        if (settings.num_points) {
            rgb_composite_medium(current,
                                 cursor_depth,
                                 boundary_depth,
                                 attenuation,
                                 backscatter,
                                 medium_colors,
                                 active,
                                 value,
                                 transmittance);
        }
        cursor_depth = boundary_depth;
        if (!finished && has_boundary) {
            current = next;
        }
        active = rgb_active_passes(
            transmittance, active, settings.transmittance_threshold);
    }

    Vec3 background_color = make_vec3(background);
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        direct_output[3 * ray_index + channel] =
            value[channel] +
            component(background_color, channel) * transmittance[channel];
        backscatter_output[3 * ray_index + channel] = value[channel + 3];
        // The six-pass path reads 1 - (1 - T) from the scalar alpha channel.
        transmission_output[3 * ray_index + channel] =
            1.0f - (1.0f - transmittance[channel]);
    }
    num_intersections[ray_index] = steps;
}

// backward_event reduced to the one channel a pass contributes to.
__device__ inline float rgb_backward_event(float alpha,
                                           float color,
                                           float final_value,
                                           float final_transmittance,
                                           float value_gradient,
                                           float alpha_gradient,
                                           float &prefix,
                                           float &transmittance,
                                           float &color_gradient) {
    alpha = fminf(fmaxf(alpha, 0.0f), 1.0f - 1.0e-7f);
    float transmittance_before = transmittance;
    float weight = transmittance_before * alpha;
    color_gradient = value_gradient * weight;
    float prefix_after = prefix + color * weight;
    float transmittance_after = transmittance_before * (1.0f - alpha);
    float tail = (final_value - prefix_after) *
                 (1.0f / fmaxf(transmittance_after, kEpsilon));
    float alpha_derivative =
        transmittance_before * (value_gradient * (color - tail)) +
        alpha_gradient * final_transmittance / fmaxf(1.0f - alpha, kEpsilon);
    prefix = prefix_after;
    transmittance = transmittance_after;
    return alpha_derivative;
}

struct RgbBackwardState {
    float final_value[kRgbPasses];
    float final_transmittance[kRgbPasses];
    float value_gradient[kRgbPasses];
    float alpha_gradient[kRgbPasses];
    float prefix[kRgbPasses];
    float transmittance[kRgbPasses];
};

// accumulate_gaussian_geometry in reverse mode: the same function as
// evaluate_gaussian_dual (plane depth from the formula with the saved depth as
// its value, r^2 clamped without derivative below epsilon, exp(kappa log r^2)),
// differentiated by hand instead of carrying 11 dual derivatives, which needs
// far fewer registers and operations.
__device__ inline void accumulate_gaussian_geometry_reverse(
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
    float *raw_kappa_grad,
    Vec3 screen_right,
    Vec3 screen_down,
    float *means_screen_abs) {
    Vec3 mean = make_vec3(means + 3 * index);
    const float *q = quaternions + 4 * index;
    float w = q[0], x = q[1], y = q[2], z = q[3];
    float factor = 2.0f / (w * w + x * x + y * y + z * z);
    // Rotation entries r_ij = delta_ij + factor * p_ij(q).
    float p[3][3] = {{-(y * y + z * z), x * y + w * z, x * z - w * y},
                     {x * y - w * z, -(x * x + z * z), y * z + w * x},
                     {x * z + w * y, y * z - w * x, -(x * x + y * y)}};
    Vec3 row0 = {1.0f + factor * p[0][0], factor * p[0][1], factor * p[0][2]};
    Vec3 row1 = {factor * p[1][0], 1.0f + factor * p[1][1], factor * p[1][2]};
    Vec3 row2 = {factor * p[2][0], factor * p[2][1], 1.0f + factor * p[2][2]};

    Vec3 offset = mean - origin;
    float numerator = dot(row2, offset);
    float denominator = dot(row2, direction);
    float formula_depth = numerator / denominator;
    Vec3 local = origin + direction * hit_depth - mean;
    float inverse_scale_x = expf(-log_scales[2 * index]);
    float inverse_scale_y = expf(-log_scales[2 * index + 1]);
    float projection_u = dot(row0, local);
    float projection_v = dot(row1, local);
    float u = inverse_scale_x * projection_u;
    float v = inverse_scale_y * projection_v;
    float radius_squared = u * u + v * v;
    bool clamped = !(radius_squared >= kEpsilon);
    if (clamped) {
        radius_squared = kEpsilon;
    }
    float exp_raw = expf(raw_kappa[index]);
    float kappa = 1.0f + logf(1.0f + exp_raw);
    float log_radius = logf(radius_squared);
    float radial_power = expf(kappa * log_radius);
    float exponent = -0.5f * radial_power / kappa;
    float response = expf(exponent);
    float opacity = 1.0f / (1.0f + expf(-opacity_logits[index]));

    // Reverse pass.
    float opacity_logit_gradient =
        alpha_coefficient * response * opacity * (1.0f - opacity);
    float exponent_gradient = alpha_coefficient * opacity * response;
    float power_gradient = exponent_gradient * -0.5f / kappa;
    float kappa_gradient = -exponent_gradient * exponent / kappa;
    float product_gradient = power_gradient * radial_power;
    kappa_gradient += product_gradient * log_radius;
    float radius_gradient = clamped ? 0.0f : product_gradient * kappa / radius_squared;
    float u_gradient = radius_gradient * 2.0f * u;
    float v_gradient = radius_gradient * 2.0f * v;
    float log_scale_x_gradient = -u_gradient * projection_u * inverse_scale_x;
    float log_scale_y_gradient = -v_gradient * projection_v * inverse_scale_y;
    float projection_u_gradient = u_gradient * inverse_scale_x;
    float projection_v_gradient = v_gradient * inverse_scale_y;
    Vec3 row0_gradient = local * projection_u_gradient;
    Vec3 row1_gradient = local * projection_v_gradient;
    Vec3 local_gradient =
        row0 * projection_u_gradient + row1 * projection_v_gradient;
    float depth_gradient = dot(local_gradient, direction) + depth_coefficient;
    float numerator_gradient = depth_gradient / denominator;
    float denominator_gradient =
        -depth_gradient * formula_depth / denominator;
    Vec3 row2_gradient =
        offset * numerator_gradient + direction * denominator_gradient;
    Vec3 mean_gradient = row2 * numerator_gradient - local_gradient;

    float g[3][3] = {{row0_gradient.x, row0_gradient.y, row0_gradient.z},
                     {row1_gradient.x, row1_gradient.y, row1_gradient.z},
                     {row2_gradient.x, row2_gradient.y, row2_gradient.z}};
    float weighted = 0.0f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            weighted += g[i][j] * p[i][j];
        }
    }
    // d factor / d q_k = -factor^2 q_k.
    float shared = factor * factor * weighted;
    float w_gradient =
        factor * (g[0][1] * z - g[0][2] * y - g[1][0] * z + g[1][2] * x +
                  g[2][0] * y - g[2][1] * x) -
        shared * w;
    float x_gradient =
        factor * (g[0][1] * y + g[0][2] * z + g[1][0] * y -
                  2.0f * g[1][1] * x + g[1][2] * w + g[2][0] * z -
                  g[2][1] * w - 2.0f * g[2][2] * x) -
        shared * x;
    float y_gradient =
        factor * (-2.0f * g[0][0] * y + g[0][1] * x - g[0][2] * w +
                  g[1][0] * x + g[1][2] * z + g[2][0] * w + g[2][1] * z -
                  2.0f * g[2][2] * y) -
        shared * y;
    float z_gradient =
        factor * (-2.0f * g[0][0] * z + g[0][1] * w + g[0][2] * x -
                  g[1][0] * w - 2.0f * g[1][1] * z + g[1][2] * y +
                  g[2][0] * x + g[2][1] * y) -
        shared * z;
    float raw_kappa_gradient = kappa_gradient * exp_raw * (1.0f / (1.0f + exp_raw));

    atomic_add_vec3(means_grad + 3 * index, mean_gradient);
    if (means_screen_abs != nullptr) {
        // Absolute per-ray gradient along the screen axes (AbsGS): opposite
        // signs from different pixels do not cancel.
        atomicAdd(means_screen_abs + 2 * index,
                  fabsf(dot(mean_gradient, screen_right)));
        atomicAdd(means_screen_abs + 2 * index + 1,
                  fabsf(dot(mean_gradient, screen_down)));
    }
    atomicAdd(log_scales_grad + 2 * index, log_scale_x_gradient);
    atomicAdd(log_scales_grad + 2 * index + 1, log_scale_y_gradient);
    atomicAdd(quaternions_grad + 4 * index, w_gradient);
    atomicAdd(quaternions_grad + 4 * index + 1, x_gradient);
    atomicAdd(quaternions_grad + 4 * index + 2, y_gradient);
    atomicAdd(quaternions_grad + 4 * index + 3, z_gradient);
    atomicAdd(opacity_logits_grad + index, opacity_logit_gradient);
    atomicAdd(raw_kappa_grad + index, raw_kappa_gradient);
}

// Backward writes few global atomics: parameter gradients are summed in
// registers while they keep going to the same address and written once.
//
// CellGradient: attenuation, backscatter and colour gradients of the cell the
// ray is in, written when the ray leaves it.
struct CellGradient {
    uint32_t cell;
    float density[kRgbPasses];
    float color[3];
};

__device__ inline void flush_cell(CellGradient &gradient,
                                  float *attenuation_grad,
                                  float *backscatter_grad,
                                  float *medium_colors_grad) {
    if (gradient.cell == UINT32_MAX) {
        return;
    }
    uint32_t cell = gradient.cell;
#pragma unroll
    for (int pass = 0; pass < kRgbPasses; ++pass) {
        if (gradient.density[pass] != 0.0f) {
            atomicAdd(pass < 3 ? attenuation_grad + 3 * cell + pass
                               : backscatter_grad + 3 * cell + pass - 3,
                      gradient.density[pass]);
        }
        gradient.density[pass] = 0.0f;
    }
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        if (gradient.color[channel] != 0.0f) {
            atomicAdd(medium_colors_grad + 3 * cell + channel,
                      gradient.color[channel]);
        }
        gradient.color[channel] = 0.0f;
    }
    gradient.cell = UINT32_MAX;
}

// Backward of one medium segment for every active pass; returns the summed
// derivative with respect to the segment length (the caller handles its
// endpoints).
__device__ inline float rgb_backward_medium(uint32_t cell,
                                            float start,
                                            float end,
                                            uint32_t active,
                                            const float *attenuation,
                                            const float *backscatter,
                                            const float *medium_colors,
                                            float *attenuation_grad,
                                            float *backscatter_grad,
                                            float *medium_colors_grad,
                                            CellGradient &cell_gradient,
                                            RgbBackwardState &state) {
    float length = fmaxf(end - start, 0.0f);
    if (length <= 0.0f) {
        return 0.0f;
    }
    if (cell_gradient.cell != cell) {
        flush_cell(cell_gradient,
                   attenuation_grad,
                   backscatter_grad,
                   medium_colors_grad);
        cell_gradient.cell = cell;
    }
    Vec3 color = make_vec3(medium_colors + 3 * cell);
    float length_derivative = 0.0f;
#pragma unroll
    for (int pass = 0; pass < kRgbPasses; ++pass) {
        if (!(active & (1u << pass))) {
            continue;
        }
        float raw_density =
            rgb_pass_density(attenuation, backscatter, cell, pass);
        float density = fmaxf(raw_density, 0.0f);
        float one_minus_alpha = expf(-density * length);
        float alpha = 1.0f - one_minus_alpha;
        float color_gradient;
        float alpha_derivative = rgb_backward_event(
            alpha,
            pass < 3 ? 0.0f : component(color, pass - 3),
            state.final_value[pass],
            state.final_transmittance[pass],
            state.value_gradient[pass],
            state.alpha_gradient[pass],
            state.prefix[pass],
            state.transmittance[pass],
            color_gradient);
        if (pass >= 3) {
            cell_gradient.color[pass - 3] += color_gradient;
        }
        if (raw_density > 0.0f) {
            cell_gradient.density[pass] +=
                alpha_derivative * length * one_minus_alpha;
        }
        length_derivative += alpha_derivative * density * one_minus_alpha;
    }
    return length_derivative;
}

// The start of the next medium segment: a Gaussian event or a cell boundary.
// Its gradient has up to three parts at the same depth (end of the segment
// before it, its own alpha for a Gaussian, start of the segment after it); they
// are summed and written with one evaluation.
struct PendingEndpoint {
    EndpointOwner owner;
    float alpha_coefficient;
    float depth_coefficient;
};

// Site gradient of the boundary just written for its second site: the next
// boundary's first site is usually the same one.
struct SiteCarry {
    uint32_t site;
    Vec3 gradient;
};

__device__ inline void flush_site_carry(SiteCarry &carry, float *points_grad) {
    if (carry.site != UINT32_MAX) {
        atomic_add_vec3(points_grad + 3 * carry.site, carry.gradient);
        carry.site = UINT32_MAX;
    }
}

__device__ inline void flush_endpoint(PendingEndpoint &pending,
                                      SiteCarry &carry,
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
                                      float *raw_kappa_grad,
                                      Vec3 screen_right,
                                      Vec3 screen_down,
                                      float *means_screen_abs) {
    const EndpointOwner &owner = pending.owner;
    if (owner.type == OwnerType::Gaussian) {
        accumulate_gaussian_geometry_reverse(owner.gaussian,
                                     owner.depth,
                                     pending.alpha_coefficient,
                                     pending.depth_coefficient,
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
                                     raw_kappa_grad,
                                     screen_right,
                                     screen_down,
                                     means_screen_abs);
    } else if (owner.type == OwnerType::Boundary &&
               pending.depth_coefficient != 0.0f) {
        Vec3 first = make_vec3(points + 3 * owner.first);
        Vec3 second = make_vec3(points + 3 * owner.second);
        Vec3 first_gradient =
            cell_intersection_gradient(first, second, origin, direction) *
            pending.depth_coefficient;
        Vec3 second_gradient =
            cell_intersection_gradient(second, first, origin, direction) *
            pending.depth_coefficient;
        if (carry.site == owner.first) {
            first_gradient = first_gradient + carry.gradient;
            carry.site = UINT32_MAX;
        }
        flush_site_carry(carry, points_grad);
        atomic_add_vec3(points_grad + 3 * owner.first, first_gradient);
        carry = {owner.second, second_gradient};
    }
    pending = {{OwnerType::None, -1, 0, 0, 0.0f}, 0.0f, 0.0f};
}

__global__ void rgb_backward_kernel(
    HybridLaunchSettings settings,
    const float *points,
    const float *attenuation,
    const float *backscatter,
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
    const float *direct_output,
    const float *backscatter_output,
    const float *transmission_output,
    const float *direct_output_grad,
    const float *backscatter_output_grad,
    const float *transmission_output_grad,
    float *points_grad,
    float *attenuation_grad,
    float *backscatter_grad,
    float *medium_colors_grad,
    float *means_grad,
    float *log_scales_grad,
    float *quaternions_grad,
    float *gaussian_colors_grad,
    float *opacity_logits_grad,
    float *raw_kappa_grad,
    const float *screen_axes,
    float *means_screen_abs) {
    uint32_t ray_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_index >= settings.num_rays) {
        return;
    }
    Vec3 origin = make_vec3(rays + 6 * ray_index);
    Vec3 direction = normalized(make_vec3(rays + 6 * ray_index + 3));
    Vec3 screen_right = {0.0f, 0.0f, 0.0f};
    Vec3 screen_down = {0.0f, 0.0f, 0.0f};
    if (means_screen_abs != nullptr) {
        screen_right = make_vec3(screen_axes);
        screen_down = make_vec3(screen_axes + 3);
    }
    RgbBackwardState state;
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        // Direct pass: its alpha output is 1 - transmission.
        state.final_value[channel] = direct_output[3 * ray_index + channel];
        state.final_transmittance[channel] =
            transmission_output[3 * ray_index + channel];
        state.value_gradient[channel] =
            direct_output_grad[3 * ray_index + channel];
        state.alpha_gradient[channel] =
            -transmission_output_grad[3 * ray_index + channel];
        // Backscatter pass: its alpha output is unused.
        state.final_value[channel + 3] =
            backscatter_output[3 * ray_index + channel];
        state.final_transmittance[channel + 3] = 0.0f;
        state.value_gradient[channel + 3] =
            backscatter_output_grad[3 * ray_index + channel];
        state.alpha_gradient[channel + 3] = 0.0f;
    }
    CellGradient cell_gradient;
    cell_gradient.cell = UINT32_MAX;
#pragma unroll
    for (int pass = 0; pass < kRgbPasses; ++pass) {
        state.prefix[pass] = 0.0f;
        state.transmittance[pass] = 1.0f;
        cell_gradient.density[pass] = 0.0f;
    }
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        cell_gradient.color[channel] = 0.0f;
    }
    uint32_t active = rgb_active_passes(
        state.transmittance, kAllRgbPasses, settings.transmittance_threshold);
    float cursor_depth = settings.near_plane;
    PendingEndpoint pending{
        {OwnerType::None, -1, 0, 0, cursor_depth}, 0.0f, 0.0f};
    SiteCarry carry{UINT32_MAX, {0.0f, 0.0f, 0.0f}};
    HitCursor hit_cursor{0};
    uint32_t steps = 0;
    uint32_t current = settings.num_points ? start_points[ray_index] : 0;
    bool finished = false;

#define FOAMSURF_FLUSH_ENDPOINT()                                              \
    flush_endpoint(pending, carry, origin, direction, points, means,          \
                   log_scales, quaternions, opacity_logits, raw_kappa,         \
                   points_grad, means_grad, log_scales_grad, quaternions_grad, \
                   opacity_logits_grad, raw_kappa_grad, screen_right,          \
                   screen_down, means_screen_abs)

    while (!finished && cursor_depth < settings.far_plane && active &&
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
            float length_derivative = 0.0f;
            if (settings.num_points) {
                length_derivative = rgb_backward_medium(current,
                                                        cursor_depth,
                                                        hit_depth,
                                                        active,
                                                        attenuation,
                                                        backscatter,
                                                        medium_colors,
                                                        attenuation_grad,
                                                        backscatter_grad,
                                                        medium_colors_grad,
                                                        cell_gradient,
                                                        state);
                pending.depth_coefficient -= length_derivative;
            }
            FOAMSURF_FLUSH_ENDPOINT();
            pending = {{OwnerType::Gaussian, gaussian_index, 0, 0, hit_depth},
                       0.0f,
                       length_derivative};
            float alpha = evaluate_gaussian_scalar(gaussian_index,
                                                   hit_depth,
                                                   origin,
                                                   direction,
                                                   means,
                                                   log_scales,
                                                   quaternions,
                                                   opacity_logits,
                                                   raw_kappa)
                              .alpha;
            Vec3 color = make_vec3(gaussian_colors + 3 * gaussian_index);
            float color_gradient[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int pass = 0; pass < kRgbPasses; ++pass) {
                if (!(active & (1u << pass))) {
                    continue;
                }
                float pass_color_gradient;
                pending.alpha_coefficient += rgb_backward_event(
                    alpha,
                    pass < 3 ? component(color, pass) : 0.0f,
                    state.final_value[pass],
                    state.final_transmittance[pass],
                    state.value_gradient[pass],
                    state.alpha_gradient[pass],
                    state.prefix[pass],
                    state.transmittance[pass],
                    pass_color_gradient);
                if (pass < 3) {
                    color_gradient[pass] = pass_color_gradient;
                }
            }
#pragma unroll
            for (int channel = 0; channel < 3; ++channel) {
                if (color_gradient[channel] != 0.0f) {
                    atomicAdd(gaussian_colors_grad + 3 * gaussian_index + channel,
                              color_gradient[channel]);
                }
            }
            active = rgb_active_passes(
                state.transmittance, active, settings.transmittance_threshold);
            cursor_depth = fmaxf(cursor_depth, hit_depth);
            ++hit_cursor.slot;
            if (!active) {
                break;
            }
        }
        if (!active) {
            break;
        }
        float length_derivative = 0.0f;
        if (settings.num_points) {
            length_derivative = rgb_backward_medium(current,
                                                    cursor_depth,
                                                    boundary_depth,
                                                    active,
                                                    attenuation,
                                                    backscatter,
                                                    medium_colors,
                                                    attenuation_grad,
                                                    backscatter_grad,
                                                    medium_colors_grad,
                                                    cell_gradient,
                                                    state);
            pending.depth_coefficient -= length_derivative;
        }
        FOAMSURF_FLUSH_ENDPOINT();
        pending = {boundary_owner, 0.0f, length_derivative};
        cursor_depth = boundary_depth;
        if (!finished && has_boundary) {
            current = next;
        }
        active = rgb_active_passes(
            state.transmittance, active, settings.transmittance_threshold);
    }
    FOAMSURF_FLUSH_ENDPOINT();
#undef FOAMSURF_FLUSH_ENDPOINT
    flush_site_carry(carry, points_grad);
    flush_cell(cell_gradient, attenuation_grad, backscatter_grad,
               medium_colors_grad);
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

void surface_hit_cuda(uint32_t num_rays,
                      uint32_t hit_buffer_size,
                      float near_plane,
                      float far_plane,
                      const float *hit_depths,
                      const int32_t *hit_indices,
                      float *surface_depth,
                      int32_t *surface_index,
                      const void *stream) {
    constexpr uint32_t block_size = 256;
    uint32_t grid_size = (num_rays + block_size - 1) / block_size;
    cudaStream_t cuda_stream =
        *reinterpret_cast<const cudaStream_t *>(stream);
    surface_hit_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        num_rays, hit_buffer_size, near_plane, far_plane, hit_depths,
        hit_indices, surface_depth, surface_index);
}

void rgb_forward_cuda(
    const HybridLaunchSettings &settings,
    const RgbSceneTensors &scene,
    const float *background,
    float *direct_output,
    float *backscatter_output,
    float *transmission_output,
    uint32_t *output_num_intersections,
    const void *stream) {
    constexpr uint32_t block_size = 128;
    uint32_t grid_size = (settings.num_rays + block_size - 1) / block_size;
    cudaStream_t cuda_stream =
        *reinterpret_cast<const cudaStream_t *>(stream);
    rgb_forward_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        settings,
        scene.points,
        scene.attenuation,
        scene.backscatter,
        scene.medium_colors,
        scene.adjacency,
        scene.offsets,
        scene.rays,
        scene.start_points,
        scene.hit_depths,
        scene.hit_indices,
        scene.means,
        scene.log_scales,
        scene.quaternions,
        scene.gaussian_colors,
        scene.opacity_logits,
        scene.raw_kappa,
        background,
        direct_output,
        backscatter_output,
        transmission_output,
        output_num_intersections);
}

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
    const void *stream) {
    constexpr uint32_t block_size = 128;
    uint32_t grid_size = (settings.num_rays + block_size - 1) / block_size;
    cudaStream_t cuda_stream =
        *reinterpret_cast<const cudaStream_t *>(stream);
    rgb_backward_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        settings,
        scene.points,
        scene.attenuation,
        scene.backscatter,
        scene.medium_colors,
        scene.adjacency,
        scene.offsets,
        scene.rays,
        scene.start_points,
        scene.hit_depths,
        scene.hit_indices,
        scene.means,
        scene.log_scales,
        scene.quaternions,
        scene.gaussian_colors,
        scene.opacity_logits,
        scene.raw_kappa,
        direct_output,
        backscatter_output,
        transmission_output,
        direct_output_grad,
        backscatter_output_grad,
        transmission_output_grad,
        gradients.points,
        gradients.attenuation,
        gradients.backscatter,
        gradients.medium_colors,
        gradients.means,
        gradients.log_scales,
        gradients.quaternions,
        gradients.gaussian_colors,
        gradients.opacity_logits,
        gradients.raw_kappa,
        gradients.screen_axes,
        gradients.means_screen_abs);
}

} // namespace foamsurf
