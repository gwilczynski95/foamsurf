"""CPU-only, checkpoint-feature partitioning. No ground-truth inputs.

Output is binary: conservative SURFACE versus its complete complement.
Outer provenance and zero-contribution cells are never SURFACE.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture


@dataclass
class Partition:
    surface: np.ndarray
    info: dict
    edges: np.ndarray
    histogram: np.ndarray
    smooth: np.ndarray
    model: object = None
    center: object = None
    scale: object = None
    component: int = -1


def variants():
    """Supported production methods; historical sweep outputs remain readable."""
    return [
        dict(name='weighted_valley_w1_s2', method='valley', power=1., smoothing=2.),
        dict(name='otsu_w1', method='otsu', power=1.),
        dict(name='gmm_2', method='gmm', power=1., components=2),
    ]


def otsu_threshold(hist, edges):
    h = np.asarray(hist, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if h.ndim != 1 or edges.shape != (len(h)+1,) or not np.isfinite(h).all() or (h < 0).any() or not np.isfinite(edges).all() or not (np.diff(edges)>0).all():
        raise ValueError('Invalid histogram or bin edges')
    if h.sum() <= 0 or np.count_nonzero(h) < 2:
        return None
    p = h / h.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    mass = np.cumsum(p)[:-1]
    moment = np.cumsum(p * centers)[:-1]
    score = np.full_like(mass, -np.inf)
    valid = (mass > 1e-12) & (mass < 1-1e-12)
    score[valid] = (p * centers).sum() * mass[valid] - moment[valid]
    score[valid] = score[valid] ** 2 / (mass[valid] * (1-mass[valid]))
    # Center of a tied plateau, not its arbitrary leftmost bin.
    best = np.flatnonzero(np.isclose(score, score.max(), rtol=1e-10, atol=1e-14))
    return float(np.mean(edges[best + 1])) if len(best) else None


def classify(sigma, tv, contribution, outer, *, method, power=0., smoothing=2.,
             bins=256, components=2, seed=0, name=None):
    sigma, tv, contribution = [np.asarray(a, dtype=np.float64) for a in (sigma, tv, contribution)]
    outer = np.asarray(outer, dtype=bool)
    if sigma.ndim != 1 or not (sigma.shape == tv.shape == contribution.shape == outer.shape):
        raise ValueError('Expected equally sized one-dimensional arrays')
    if any(not np.isfinite(a).all() or (a < 0).any() for a in (sigma, tv, contribution)):
        raise ValueError('Features must be finite and nonnegative')
    if method not in {'valley', 'otsu', 'gmm'} or not 0 <= power <= 1 or bins < 16 or smoothing <= 0:
        raise ValueError('Invalid partition settings')
    active = (~outer) & (contribution > 0)
    x = np.log10(np.maximum(tv, 1e-12))
    z = np.column_stack((np.log10(np.maximum(sigma, 1e-12)), x))
    weights = np.zeros_like(contribution)
    if active.any():
        weights[active] = (contribution[active] / contribution[active].max()) ** power
    hist, edges = np.histogram(x[active], bins=bins, weights=weights[active])
    smooth = gaussian_filter1d(hist.astype(float), smoothing)
    info = dict(name=name or method, method=method, power=power, smoothing=smoothing,
                bins=bins, status='uncertain', reason='', threshold_log10_tv=None,
                active_count=int(active.sum()), seed=seed)
    result = Partition(np.zeros(len(x), dtype=bool), info, edges, hist, smooth)
    if active.sum() < 8 or np.ptp(x[active]) < 1e-6:
        info['reason'] = 'Insufficient active samples or constant TV'; return result
    threshold = None
    if method == 'otsu':
        threshold = otsu_threshold(hist, edges)
    elif method == 'valley':
        # Padding allows a genuine mode at either histogram boundary.
        peaks, properties = find_peaks(np.r_[0., smooth, 0.],
            prominence=max(smooth.max()*.03, 1e-15), distance=max(2, int(2*smoothing)))
        peaks = peaks - 1
        # Select the rightmost prominent mode and its closest prominent neighbor.
        # This is an explicit high-TV-surface assumption, not semantic proof.
        info['peaks_log10_tv'] = [float((edges[p]+edges[p+1])/2) for p in peaks]
        if len(peaks) >= 2:
            left, right = peaks[-2:]
            valley = left + int(np.argmin(smooth[left:right+1]))
            depth = 1 - smooth[valley] / max(min(smooth[left], smooth[right]), 1e-30)
            info['valley_depth'] = float(depth)
            if left < valley < right and depth >= .15:
                threshold = float((edges[valley]+edges[valley+1])/2)
            else:
                info['reason'] = 'No sufficiently deep valley between rightmost modes'
        else:
            info['reason'] = 'Fewer than two prominent modes'
    else:
        # Contribution-weighted resampling: sklearn GMM has no sample_weight.
        # BIC describes this reproducible resampled distribution, not all sites.
        rng = np.random.default_rng(seed)
        ids = np.flatnonzero(active)
        chosen = rng.choice(ids, size=min(20000, len(ids)), replace=True,
                            p=weights[ids]/weights[ids].sum())
        sample = z[chosen]
        center = np.median(sample, axis=0)
        scale = np.maximum(np.std(sample, axis=0), 1e-3)
        sample = (sample-center)/scale
        if components != 2:
            raise ValueError('Only the validated two-component GMM is supported')
        candidates = [2]
        fits = [GaussianMixture(n_components=k, random_state=seed, n_init=3,
                    max_iter=200, reg_covar=1e-4).fit(sample) for k in candidates]
        scores = [float(m.bic(sample)) for m in fits]
        model = fits[int(np.argmin(scores))]
        info.update(bic=dict(zip(map(str,candidates),scores)), components=model.n_components,
                    converged=bool(model.converged_), fit_sample_count=len(sample))
        if model.n_components < 2 or not model.converged_:
            info['reason'] = 'Single component or nonconverged fit'; return result
        component = int(model.means_.sum(axis=1).argmax())
        # Reject incompatible ordering rather than silently calling a low-density mode surface.
        if np.any(model.means_[component] < model.means_.max(axis=0)-.25):
            info['reason'] = 'No component jointly high in sigma and TV'; return result
        result.model, result.center, result.scale, result.component = model, center, scale, component
        probability = model.predict_proba((z[active]-center)/scale)[:,component]
        result.surface[active] = probability >= .5
        info.update(status='candidate', surface_component=component,
                    component_means_log10=(model.means_*scale+center).tolist(),
                    posterior_cutoff=.5)
        return result
    if threshold is not None:
        result.surface = active & (x >= threshold)
        info.update(status='candidate', threshold_log10_tv=threshold)
    return result


def checkpoint_order(values, index):
    values, index = np.asarray(values), np.asarray(index)
    if index.shape != (len(values),) or not np.issubdtype(index.dtype, np.integer) or not np.array_equal(np.sort(index), np.arange(len(values))):
        raise ValueError('checkpoint_index must be a permutation')
    result = np.empty_like(values)
    result[index] = values
    return result
