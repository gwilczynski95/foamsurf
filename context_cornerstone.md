# CONTEXT CORNERSTONE: Project `FoamSurf`

### 1. General Overview
**FoamSurf** is a novel, real-time (>100 FPS), end-to-end differentiable 3D hybrid renderer designed specifically for scenes containing both sharp geometry and heterogeneous volumetric scattering media (e.g., underwater scenes, fog, murky environments). It combines two explicit 3D scene representations into a single unified rendering pipeline.

---

### 2. Core Architecture & Representations
* **Solid Geometry (Surfaces): 2D Gaussians / Surfels**
  * *Purpose:* Models sharp, opaque, hard surfaces (seabed, rocks, structures).
  * *Parametrization:* Planar 2D Gaussian disks defined by position ($\mu \in \mathbb{R}^3$), 2D scale and rotation ($\Sigma \in \mathbb{R}^{2\times2}$ via quaternions/vectors), base opacity ($\alpha$), and directional appearance.
  * *Key Property:* Evaluated in 3D local coordinates ($P_{hit\_prim} = P_{hit} - \mu$), allowing fully differentiable analytic gradients for position, scale, and rotation without relying on screen-space projections.
* **Scattering Medium (Volume): Radiant Foam (Sparse Voronoi Cell Graph)**
  * *Purpose:* Models soft, continuous, low-frequency volumetric scattering and absorption (water/fog).
  * *Parametrization:* Space-partitioning polyhedral Voronoi tessellation where each cell holds piecewise-constant optical density ($\sigma_{med}$) and emission/radiance ($c_{med}$).
  * *Key Property:* Ray marching uses topological adjacency (Weiler’s cell-stepping algorithm), skipping $O(\log N)$ BVH queries for the volume. Differentiable cell boundary changes occur continuously as face surface areas shrink to zero at topological flips.

---

### 3. Rendering Pipeline (Forward Pass)
1. **Pre-Pass (PyTorch):** Directional RGB colors for 2D Gaussians are pre-baked per camera pose using Spherical Harmonics (SH) to preserve VRAM bandwidth.
2. **Unified OptiX + CUDA Launch (`__raygen__` Kernel):**
   * **OptiX Hardware Acceleration (Primary Surface Culling):** Hardware RT Cores query the BVH tree containing 2D Gaussian proxies, outputting a sorted local hit buffer (`t_hit`, `gauss_index`) in thread registers.
   * **Voronoi Marching (Medium Integration):** In the same CUDA thread, the ray steps through Voronoi cells from $t_{in}$ to $t_{out}$ using face normals.
   * **Intersection Fusion:** As the ray steps through a cell, it checks `t_hit` from the OptiX buffer. If $t_{hit} \in [t_{in}, t_{out}]$, it computes:
     1. Medium integration from $t_{in}$ to $t_{hit}$ (Beer-Lambert law: $\alpha_{med} = 1 - e^{-\sigma_{med} \Delta t}$).
     2. Hard surface blending from the 2D Gaussian at $t_{hit}$ ($\alpha_g, c_g$).
     3. Medium integration from $t_{hit}$ to $t_{out}$.
   * **Early Ray Termination:** As Gaussians are constrained to be opaque ($\alpha_g \approx 1.0$), transmittance $T$ drops rapidly below a threshold ($T_{threshold}$), terminating the ray immediately and ensuring the paged OptiX launch executes in a single pass (avoiding multi-pass depth-peeling iterations).

---

### 4. Backward Pass & Autograd Integration
* **Saved Hit-Buffer Re-marching (No Backward OptiX):**
  * `forward` saves `hit_buffer` (`[gauss_idx, t_hit]`), camera rays, and Voronoi cell states into `torch.autograd.Function` context (`ctx`).
  * `backward` executes a pure CUDA re-marching kernel using the paged `hit_buffer` coordinates without calling OptiX.
* **Gradient Flow:**
  * Computes exact analytical chain-rule gradients for 2D Gaussians ($\frac{\partial L}{\partial \mu}, \frac{\partial L}{\partial \Sigma}, \frac{\partial L}{\partial \alpha_g}, \frac{\partial L}{\partial c_g}$).
  * Computes exact gradients for Voronoi cells ($\frac{\partial L}{\partial \sigma_{med}}, \frac{\partial L}{\partial c_{med}}$) and Voronoi site positions via boundary displacement gradients ($\frac{\partial t}{\partial P_{primal}}$).

---

### 5. Entanglement Prevention (Role Separation Losses)
To prevent representations from swapping roles (e.g., Gaussians trying to model fog, or Voronoi cells clustering into hard rocks):
* **Gaussians (Hard Surface Constraints):**
  * Opacity Binarization Loss + aggressive pruning ($\alpha_g < 0.5$) forcing Gaussians to be non-transparent.
* **Voronoi Cells (Soft Volume Constraints):**
  * Total Variation (TV) / Smoothness Loss on cell densities ($\sigma_{med}$) to penalize high-frequency jumps.
  * *Quantile/Distortion Loss is explicitly DISABLED* for the Voronoi mesh to prevent cells from collapsing into flat planes.

---

### 6. Tech Stack & Reference Implementations
* **Framework:** PyTorch (`torch.autograd.Function` C++ Extensions), CUDA, NVIDIA OptiX SDK.
* **Reference Baselines & Modules:**
  * `REdiSplats` / `FALCON`: OptiX-based 2D Gaussian intersection & local-coordinate gradient derivation.
  * `Radiant Foam`: Topological Voronoi cell-stepping ray tracing algorithms and CUDA data structures.
  * `3DGUT`: Evaluation of particle responses in 3D along the ray.
  * `WaterSplatting`: Media image formation model (direct + backscatter components) and regularized loss formulations.
