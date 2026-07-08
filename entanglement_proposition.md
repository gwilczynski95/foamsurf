# FoamSurf: Surface–Medium Disentanglement Regularization Proposal

## 1. Motivation

FoamSurf uses two different scene representations:

* **2D Gaussian surfels** for hard, sharp, mostly opaque surfaces.
* **Voronoi / Radiant-Foam cells** for soft, spatially varying participating media such as water, haze, fog, turbidity, or suspended particles.

The main risk of this hybrid design is **representation entanglement**.

In an ideal solution:

* Gaussian surfels should explain **solid objects**, sharp geometry, edges, texture, and view-dependent surface appearance.
* Voronoi cells should explain **soft volumetric effects**, such as absorption, scattering, haze, and local turbidity.

However, if both representations are optimized only with an RGB reconstruction loss, they may compete for the same signal. This can lead to degenerate solutions:

* Gaussians may start modeling fog or water as semi-transparent floaters.
* Voronoi cells may collapse around object boundaries and model hard surfaces as dense volumetric sheets.
* Medium effects may be baked into surface colors.
* Surface geometry may become contaminated by soft volumetric artifacts.
* Mesh extraction from the Gaussian branch may produce noisy or physically meaningless geometry.

Therefore, the key goal of the regularization is:

> Encourage a clean separation between hard surfaces and soft participating media.

This document describes a proposed loss stack and training strategy for achieving this disentanglement.

---

## 2. Desired Representation Behavior

### 2.1 Gaussian Surface Branch

The Gaussian branch should behave like a surface representation.

It should model:

* sharp object boundaries,
* high-frequency texture,
* hard geometry,
* view-dependent appearance,
* opaque or nearly opaque surfaces,
* geometry suitable for editing and mesh extraction.

It should not model:

* semi-transparent fog,
* soft haze,
* large diffuse clouds,
* low-density volumetric effects,
* smooth attenuation along rays.

In short:

> If a Gaussian is active, it should be a hard surface element. Otherwise, it should disappear.

---

### 2.2 Voronoi Medium Branch

The Voronoi branch should behave like a spatially varying participating medium.

It should model:

* local absorption,
* scattering,
* backscatter,
* turbidity,
* smooth volumetric density changes,
* low-frequency color shifts caused by the medium.

It should not model:

* sharp surface boundaries,
* object texture,
* high-frequency edges,
* thin dense shells around geometry,
* surface-like structures.

In short:

> The medium should be smooth, spatial, low-frequency, and volumetric.

---

## 3. Notation

Let:

* ( $r$ ) be a camera ray.
* ( $I_r$ ) be the ground-truth RGB color for ray ( r ).
* ( $\hat{I}_r$ ) be the rendered RGB color.
* ( $\alpha_g$ ) be the opacity of Gaussian ( g ).
* ( $c_g$ ) be the color or appearance contribution of Gaussian ( g ).
* ( $w_g$ ) be the accumulated rendering contribution of Gaussian ( g ).
* ( $\sigma_i$ ) be the optical density of Voronoi cell ( i ).
* ( $c_i^{med}$ ) be the medium color / radiance / scattering parameter of Voronoi cell ( i ).
* ( $\Delta t_{r,i}$ ) be the length of the segment traveled by ray ( r ) inside Voronoi cell ( i ).
* ( $E$ ) be the set of neighboring Voronoi-cell pairs.
* ( $C_{med}$ ) be the rendered medium-only contribution.
* ( $C_{surf}$ ) be the rendered surface-only contribution.
* ( $HP(\cdot)$ ) be a high-pass filter, for example image minus Gaussian-blurred image.

The total loss can be written as:

$$
L = L_{rgb} * \lambda_{\alpha-bin} L_{\alpha-bin} * \lambda_{opaque} L_{opaque} * \lambda_{\tau} L_{\tau} * \lambda_{tv} L_{med-TV} * \lambda_{cap} L_{med-cap} * \lambda_{hp} L_{med-HP} * \lambda_{rep} L_{site-repulsion} * \lambda_{scale} L_{gauss-scale}
$$

Not every term needs to be enabled from the beginning. Some terms should be activated only after warm-up.

---

# 4. Reconstruction Loss

## 4.1 RGB Reconstruction Loss

The basic rendering loss is:

$$
L_{rgb} = \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}}| \hat{I}_r - I_r |_1
$$

Optionally, this can be combined with DSSIM or perceptual losses:

$$
L_{rgb} = (1 - \lambda_{ssim}) L_1 * \lambda_{ssim} L_{DSSIM}
$$

### Purpose

This loss ensures that the final rendered image matches the training views.

### Problem

On its own, this loss does not enforce any separation between surface and medium. It only cares about the final pixel color.

Therefore, without additional regularization, both branches may explain the same visual effect.

For example:

* Gaussians may model haze as transparent floaters.
* Voronoi cells may model object boundaries as dense volumetric regions.
* Surface colors may absorb medium effects.

This is why disentanglement regularization is necessary.

---

# 5. Gaussian Surface Regularization

## 5.1 Opacity Binarization Loss

We want Gaussian opacity to become close to either 0 or 1.

$$
L_{\alpha-bin} = \frac{1}{N_g} \sum_g \alpha_g (1 - \alpha_g)
$$

This function is minimized when:

* ( $\alpha_g = 0$ ), or
* ( $\alpha_g = 1$ ).

It is maximal around ( $\alpha_g = 0.5$ ).

### Purpose

This discourages semi-transparent Gaussians.

Semi-transparent Gaussians are dangerous because they can behave like fog particles. If many low-opacity Gaussians are placed in empty space, they may start modeling participating media instead of hard surfaces.

### Expected effect

This loss should push the Gaussian branch toward a binary decision:

* either a Gaussian is a real surface primitive,
* or it should be removed.

It helps prevent the Gaussian branch from becoming a volumetric cloud.

---

## 5.2 Opaque-if-Used Loss

Opacity binarization alone is not enough. A Gaussian can minimize the binarization loss by going to zero opacity, even if it is useful.

Therefore, for Gaussians that actually contribute to rendering, we encourage high opacity.

Let ( w_g ) be a contribution weight measuring how much Gaussian ( g ) affects rendered pixels. Then:

$$
L_{opaque} = \sum_g \text{stopgrad}(w_g) \left( -\log(\alpha_g + \epsilon) \right)
$$

The `stopgrad` prevents the model from reducing the loss by artificially decreasing the contribution weight.

### Purpose

This term says:

> If a Gaussian is used to explain the image, it should become opaque.

### Expected effect

This loss should make active Gaussians behave like hard surface elements.

It discourages the following failure mode:

* A Gaussian remains partially transparent.
* It contributes a little bit to many rays.
* It starts acting like a soft volumetric particle.

Instead, the model is encouraged to make useful Gaussians opaque and prune useless ones.

---

## 5.3 Low-Opacity Pruning

This is not a differentiable loss, but an optimization policy.

After some warm-up period, periodically remove Gaussians with opacity below a threshold:

$$
\alpha_g < \alpha_{prune}
$$

Example values:

```text
alpha_prune = 0.3
alpha_prune = 0.5
```

The threshold can be increased over training.

### Purpose

This removes weak semi-transparent Gaussians before they accumulate into fog-like structures.

### Expected effect

This should prevent the Gaussian branch from explaining medium effects using many small, low-opacity primitives.

The intended rule is:

> Active Gaussians should become hard surfaces. Inactive Gaussians should disappear.

---

## 5.4 Large-Splat Suppression

A dangerous failure mode is a large, low-opacity Gaussian covering a broad image region.

This can approximate smooth haze.

Therefore, we should penalize overly large Gaussians, especially when their opacity is not close to one.

One possible loss:

$$
L_{gauss-scale} = \sum_g \max(0, s_g - s_{max})^2
$$

where ( $s_g$ ) is a measure of Gaussian scale.

A stronger version weights this by transparency:

$$
L_{large-transparent} = \sum_g (1 - \alpha_g) \max(0, s_g - s_{max})^2
$$

### Purpose

This prevents Gaussians from becoming large transparent sheets or clouds.

### Expected effect

The Gaussian branch should remain local and surface-like.

It should not model global haze or smooth volumetric attenuation.

---

## 5.5 Surface Contribution High-Frequency Bias

Surfaces usually contain high-frequency visual information:

* edges,
* texture,
* object boundaries,
* specular highlights,
* normal discontinuities.

We can encourage the surface branch to explain high-frequency image content.

Let ( $HP(\cdot)$ ) be a high-pass filter. Then:

$$
L_{surf-HP} =|HP(C_{surf}) - HP(I)|_1
$$

This term should be used carefully, because the final image also contains medium effects. It is most useful on synthetic data where a clean surface-only render is available.

If a clean surface-only target ( $I_{surf}$ ) is available:

$$
L_{surf-HP} =|HP(C_{surf}) - HP(I_{surf})|_1
$$

### Purpose

This encourages sharp visual content to be handled by the Gaussian branch.

### Expected effect

Surface edges should remain in the surface representation instead of leaking into the medium branch.

---

# 6. Voronoi Medium Regularization

## 6.1 Optical Thickness Sparsity

Instead of penalizing density alone, we penalize accumulated optical thickness along rays.

For a ray ( $r$ ), define:

$$
\tau_r = \sum_{i \in r}\sigma_i \Delta t_{r,i}
$$

Then:

$$
L_{\tau} = \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \tau_r
$$

### Purpose

This encourages the medium to stay low-density unless the image reconstruction requires non-zero absorption or scattering.

### Why optical thickness instead of raw density?

A raw density penalty ignores how much of the ray travels through a cell.

For participating media, what matters physically is not just density, but density integrated over path length.

A small density over a long path can have a large visual effect.

### Expected effect

The Voronoi branch should not explain everything by increasing density everywhere.

It should only introduce medium where needed.

This helps enforce:

> The default state of the medium is empty or weakly participating.

---

## 6.2 Medium Density Cap

We do not want Voronoi cells to become extremely dense, because very dense cells can behave like opaque surfaces.

A soft cap can be used:

$$
L_{med-cap} = \sum_i \max(0, \sigma_i - \sigma_{max})^2
$$

Alternatively, cap the optical thickness of each ray-cell segment:

$$
L_{seg-cap} = \sum_{r,i} \max(0, \sigma_i \Delta t_{r,i} - \tau_{max})^2
$$

### Purpose

This prevents Voronoi cells from forming dense surface-like barriers.

### Expected effect

The medium can become locally stronger, but it should not become a hard opaque shell.

This is especially important near object boundaries, where the medium branch may otherwise try to imitate surfaces.

---

## 6.3 Graph Total Variation Smoothness

Participating media should usually be spatially smooth.

Let ( $E$ ) be the set of neighboring Voronoi cells. Then:

$$
L_{med-TV} = \sum_{(i,j) \in E}w_{ij}|\sigma_i - \sigma_j|
$$

A Huber version may be more stable:

$$
L_{med-TV} = \sum_{(i,j) \in E} w_{ij} \text{Huber}(\sigma_i - \sigma_j)
$$

The weights ( $w_{ij}$ ) may depend on face area, distance between cell centers, or simply be set to 1.

### Purpose

This discourages sharp density jumps between neighboring cells.

### Expected effect

The Voronoi branch should model smooth clouds, haze, and turbidity, not high-frequency surface boundaries.

This is one of the most important losses for preventing Voronoi cells from becoming a surface representation.

---

## 6.4 Medium High-Pass Penalty

Medium should not contain sharp high-frequency image details.

Let ( $C_{med}$ ) be the medium-only rendered contribution. Penalize its high-frequency content:

$$
L_{med-HP} =|HP(C_{med})|_1
$$

Here, ( $HP(X)$ ) can be implemented as:

$$
HP(X) = X - \text{GaussianBlur}(X)
$$

### Purpose

This explicitly tells the model:

> The medium branch should not explain edges, texture, or sharp object boundaries.

### Expected effect

The medium contribution should remain smooth in image space.

This helps avoid a common failure mode where the medium learns object silhouettes or texture-like patterns.

---

## 6.5 Voronoi Site Repulsion

If Voronoi sites are optimized freely, they may collapse onto surfaces.

This is dangerous because a cluster of small Voronoi cells can approximate a sharp boundary.

We can discourage site collapse with a repulsion loss.

For neighboring or nearby sites ( $p_i$, $p_j$ ):

$$
L_{site-repulsion} = \sum_{i,j}\max(0, d_{min} - |p_i - p_j|)^2
$$

This can be computed only for nearby pairs for efficiency.

### Purpose

This prevents Voronoi sites from clustering too strongly.

### Expected effect

The Voronoi graph should remain a spatial medium discretization rather than collapsing into thin sheets around surfaces.

---

## 6.6 Cell Volume Regularization

Another way to prevent collapse is to penalize very small cells.

If ( $V_i$ ) is the volume of cell ( $i$ ), then:

$$
L_{cell-volume} = \sum_i\max(0, V_{min} - V_i)^2
$$

### Purpose

This discourages degenerate cells.

### Expected effect

The medium representation remains volumetric, not surface-like.

This loss may be difficult to compute exactly, depending on the Voronoi implementation. A practical approximation may be based on local site distances.

---

# 7. Cross-Branch Disentanglement Losses

## 7.1 Frequency Routing Principle

The main idea is:

* high-frequency information should go to Gaussians,
* low-frequency volumetric residual should go to Voronoi.

This matches the physical nature of the scene:

* surfaces create sharp boundaries and detailed texture,
* media create smooth attenuation and scattering.

Therefore, we want to discourage:

* high-frequency content in the medium branch,
* soft low-opacity volumetric behavior in the Gaussian branch.

The most important practical term is:

$$
L_{med-HP} = |HP(C_{med})|_1
$$

This loss penalizes sharp detail in the medium rendering.

### Expected effect

If the model sees a sharp edge, it should prefer explaining it using Gaussians rather than Voronoi density.

If the model sees smooth haze, it should prefer explaining it using Voronoi rather than transparent Gaussian floaters.

---

## 7.2 Synthetic Decomposition Supervision

On synthetic datasets, we can render additional ground-truth passes:

* clean surface-only RGB,
* medium-only contribution,
* transmittance map,
* depth,
* object masks.

If available, these can be used to explicitly supervise decomposition.

For example, if ( $I_{surf}^{GT}$ ) is the clean surface-only render:

$$
L_{surf-clean} =|C_{surf} - I_{surf}^{GT}|_1
$$

If ( $T^{GT}$ ) is the ground-truth transmittance map:

$$
L_T =|T - T^{GT}|_1
$$

If ( $C_{med}^{GT}$ ) is available:

$$
L_{med} = |C_{med} - C_{med}^{GT}|_1
$$

### Purpose

Synthetic decomposition supervision can verify that the intended representation split is actually happening.

### Expected effect

This can strongly stabilize early experiments.

It should not necessarily be required for real data, but it is very useful for debugging and for proving the concept.

---

# 8. Proposed Total Loss

A practical initial version could be:

$$
L = L_{rgb} * \lambda_{\alpha-bin} L_{\alpha-bin} * \lambda_{opaque} L_{opaque} * \lambda_{\tau} L_{\tau} * \lambda_{tv} L_{med-TV} * \lambda_{cap} L_{med-cap} * \lambda_{hp} L_{med-HP} * \lambda_{rep} L_{site-repulsion} * \lambda_{scale} L_{gauss-scale}
$$

Optional synthetic terms:

$$
+ \lambda_{surf-clean} L_{surf-clean} + \lambda_T L_T + \lambda_{med} L_{med}
$$

The exact weights should be scheduled during training.

The goal is not to make every regularizer strong from the beginning. The goal is to guide the optimization so that each branch specializes in the correct type of signal.

---

# 9. Training Strategy

The training process is as important as the losses.

If both branches are fully active from the first iteration, the model may choose an easy but entangled solution.

Therefore, training should be staged.

---

## 9.1 Stage 0: Initialization

Initialize:

* Gaussians from COLMAP / SfM points / surface prior / random surface samples.
* Voronoi sites as a coarse spatial distribution covering the scene volume.
* Medium density close to zero.
* Medium color/scattering parameters to weak neutral values.

Initial medium should be weak.

The model should not start by explaining the entire scene as volume.

---

## 9.2 Stage 1: Surface Warm-Up

In the first stage, train mainly the Gaussian branch.

The medium branch is either:

* disabled,
* frozen,
* or heavily regularized toward zero density.

Use:

* ( $L_{rgb}$ ),
* ( $L_{\alpha-bin}$ ),
* ( $L_{opaque}$ ),
* pruning,
* scale regularization.

Keep medium losses strong:

* high ( $\lambda_{\tau}$ ),
* high ( $\lambda_{tv}$ ),
* high ( $\lambda_{hp}$ ).

### Purpose

The Gaussian branch should first learn the hard geometry and appearance.

### Expected effect

The model obtains a reasonable surface reconstruction before the medium is allowed to explain residuals.

This reduces the chance that the medium branch learns object boundaries.

---

## 9.3 Stage 2: Medium Residual Learning

After the surface branch captures the main geometry, gradually allow the Voronoi branch to explain remaining low-frequency residuals.

During this stage:

* reduce ( $\lambda_{\tau}$ ) slowly,
* keep ( $\lambda_{tv}$ ) active,
* keep ( $L_{med-HP}$ ) active,
* keep Gaussian pruning active,
* keep opacity binarization active.

### Purpose

The medium should learn what the surface branch cannot explain well:

* haze,
* attenuation,
* smooth scattering,
* local turbidity,
* soft volumetric color shifts.

### Expected effect

The medium branch should begin to model soft residual effects without stealing sharp geometry from the surface branch.

---

## 9.4 Stage 3: Joint Fine-Tuning

Finally, optimize both branches jointly.

Use all relevant losses:

* RGB reconstruction,
* Gaussian opacity binarization,
* opaque-if-used,
* medium optical thickness sparsity,
* medium TV,
* medium high-pass penalty,
* density cap,
* site repulsion,
* pruning.

The weights may be reduced but should not be removed entirely.

### Purpose

This stage refines the interaction between surfaces and medium.

### Expected effect

The final renderer should produce high-quality images while preserving surface-medium separation.

---

## 9.5 Stage 4: Optional Editing / Mesh Validation

After training, evaluate whether disentanglement worked.

Possible tests:

### Surface editing

Edit only the Gaussian branch:

* delete object,
* move object,
* recolor object,
* stylize object.

The medium should remain spatially stable.

### Mesh extraction

Extract a mesh only from the Gaussian branch.

The mesh should contain hard surfaces but not haze, clouds, or fog-like artifacts.

### Medium editing

Edit only the Voronoi branch:

* increase local turbidity,
* move a cloud,
* change medium color,
* reduce scattering.

The hard surfaces should remain stable.

These tests are not only demos. They directly test whether disentanglement succeeded.

---

# 10. Recommended Ablations

To verify that the regularization stack is necessary, run ablations.

## 10.1 No Disentanglement Losses

Train with only RGB loss.

Expected result:

* Gaussians and Voronoi compete.
* Medium may learn surfaces.
* Gaussians may create fog-like floaters.
* Editing and mesh extraction may become poor.

---

## 10.2 No Gaussian Opacity Binarization

Remove ( $L_{\alpha-bin}$ ).

Expected result:

* More semi-transparent Gaussians.
* Higher chance of Gaussian fog.
* More floaters.
* Worse mesh extraction.

---

## 10.3 No Opaque-if-Used Loss

Remove ( $L_{opaque}$ ).

Expected result:

* Useful Gaussians may remain semi-transparent.
* Surface branch becomes less hard.
* Medium/surface separation becomes weaker.

---

## 10.4 No Medium TV

Remove ( $L_{med-TV}$ ).

Expected result:

* Medium density becomes high-frequency.
* Voronoi cells may capture object boundaries.
* Medium branch may become surface-like.

---

## 10.5 No Medium High-Pass Penalty

Remove ( $L_{med-HP}$ ).

Expected result:

* Medium may contain sharp image details.
* Surface edges may leak into the medium branch.
* Worse interpretability of decomposition.

---

## 10.6 No Site Repulsion / Anti-Collapse

Remove ( $L_{site-repulsion}$ ).

Expected result:

* Voronoi sites may cluster near surfaces.
* Medium representation may collapse into thin sheets.
* Voronoi branch may start modeling geometry.

---

## 10.7 No Training Curriculum

Train both branches jointly from the start.

Expected result:

* Optimization may choose an entangled solution.
* Early medium branch may steal surface signal.
* Surface reconstruction may become weaker.

---

# 11. Practical Implementation Notes

## 11.1 Contribution Weights for Gaussians

For ( $L_{opaque}$ ), ( $w_g$ ) can be estimated as:

* accumulated alpha contribution,
* number of rays hit by the Gaussian,
* total transmittance-weighted contribution,
* gradient magnitude,
* rendered visibility count.

A simple version:

$$
w_g = \sum_{r} T_{r,g} \alpha_g
$$

where ( $T_{r,g}$ ) is the accumulated transmittance before Gaussian ( $g$ ) along ray ( $r$ ).

Normalize or clamp ( $w_g$ ) for stability.

---

## 11.2 High-Pass Filter

A simple implementation:

```python
low = gaussian_blur(image, kernel_size=K, sigma=sigma)
high = image - low
loss = high.abs().mean()
```

For the medium branch:

```python
L_med_HP = high_pass(C_med).abs().mean()
```

This penalizes high-frequency medium contribution.

---

## 11.3 Optical Thickness

During Voronoi ray marching, accumulate:

```python
tau_ray += sigma_cell * delta_t
```

Then:

```python
L_tau = tau_ray.mean()
```

This is easy to compute because the renderer already traverses cells along each ray.

---

## 11.4 Graph TV

For every neighboring pair of Voronoi cells:

```python
L_tv += weight_ij * huber(sigma_i - sigma_j)
```

The neighborhood graph is naturally available from the Voronoi adjacency structure.

---

## 11.5 Pruning

A possible pruning policy:

```text
Every K iterations after warm-up:
    remove Gaussian g if:
        alpha_g < alpha_threshold
        and contribution_g < contribution_threshold
```

This avoids pruning useful but temporarily low-opacity Gaussians too early.

---

# 12. Summary

The goal of this regularization stack is to prevent surface-medium entanglement.

The desired behavior is:

| Component           | Should model                                   | Should not model                          |
| ------------------- | ---------------------------------------------- | ----------------------------------------- |
| 2D Gaussian surfels | hard surfaces, edges, texture, geometry        | fog, haze, smooth volumetric density      |
| Voronoi cells       | soft medium, scattering, absorption, turbidity | object boundaries, texture, hard geometry |

The core strategy is:

1. Make active Gaussians opaque and surface-like.
2. Remove weak transparent Gaussians.
3. Keep Voronoi density smooth, sparse, and low-frequency.
4. Prevent Voronoi sites from collapsing into surfaces.
5. Route high-frequency image content to Gaussians.
6. Route low-frequency volumetric residuals to Voronoi.
7. Train in stages: surface first, medium second, joint fine-tuning last.

If successful, this disentanglement should enable:

* better rendering of heterogeneous participating media,
* cleaner hard-surface reconstruction,
* more meaningful scene editing,
* cleaner mesh extraction,
* possible physics simulation on extracted hard geometry,
* stable medium editing independent of surface appearance.

The main hypothesis is:

> A hybrid renderer can outperform single-representation methods when the scene contains both sharp opaque surfaces and spatially varying participating media, but only if the optimization is regularized to preserve a clean surface-medium separation.
