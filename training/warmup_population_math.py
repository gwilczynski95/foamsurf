"""Pure Torch policy primitives, independently testable on CPU."""
import torch


def population_masks(contribution, radii, adjacency, offsets, threshold, radius):
    active = contribution > threshold
    degree = offsets[1:].long() - offsets[:-1].long()
    owners = torch.repeat_interleave(torch.arange(len(active), device=active.device), degree)
    neighbors = torch.zeros(len(active), dtype=torch.long, device=active.device)
    neighbors.index_add_(0, owners, active[adjacency.long()].long())
    # Correct CSR segment sum, including the first edge of every cell.
    return (~active) & (neighbors == 0) & (radii < radius)


def sample_parents(error, radii, requested):
    weights = error.clamp_min(0) * radii
    if not bool(torch.isfinite(weights).all()):
        raise FloatingPointError("non-finite population sampling weights")
    count = min(requested, int((weights > 0).sum()))
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=weights.device)
    return torch.multinomial(weights / weights.max(), count, replacement=False)


def offspring_sites(points, farthest, parents):
    direction = .25 * (points[farthest[parents]] - points[parents])
    noise = torch.randn_like(direction)
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return points[parents] + direction + .1 * direction.norm(dim=-1, keepdim=True) * noise


def growth_interval(count, initial_count, target, factor, start, horizon, minimum=100):
    if target <= initial_count or horizon <= start:
        raise ValueError("growth target/horizon must exceed initialization/start")
    return max(minimum, int((factor - 1) * count * (horizon - start) / (target - initial_count)))
