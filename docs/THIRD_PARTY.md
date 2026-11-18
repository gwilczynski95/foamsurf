# Attribution and distribution status

The repository root [LICENSE](../LICENSE) contains Apache-2.0. Third-party components,
datasets, model weights and assets retain their own terms; the root license does not
replace those terms.

| Component | Source or provenance | License information in this checkout |
|---|---|---|
| PPISP | Vendored in `third_party/ppisp` | [Apache-2.0 text](../third_party/ppisp/LICENSE) |
| Eigen | Git submodule from `https://gitlab.com/libeigen/eigen.git` | [Upstream license inventory](../external/submodules/eigen/COPYING.README); preserve its file-specific notices |
| FlaRe-base OptiX renderer | Branch `origin/generalized-depth-regularization-compound`, commit `5510485` | [Import provenance](../renderer/third_party/flare_renderer/README.md); upstream license/notice has not yet been included here |
| NVIDIA OptiX SDK | Downloaded separately by the user | NVIDIA SDK terms; not included in this repository |
| CLIP and Warp | Optional dependencies in `requirements-apps.txt` | Installed from upstream; retain their upstream license notices |
| SeaThru-NeRF | [Original project and dataset](https://sea-thru-nerf.github.io/) | Obtain from the authors under the dataset's terms |
| Insertion meshes and style images | Separate asset package | Per-asset sources, licenses and attribution still need to be supplied |

Before public distribution, resolve the missing FlaRe-base license and audit inherited
source notices, including any 3D Gaussian Splatting-derived code. An upstream research
license must not be replaced by the repository's Apache-2.0 text. The exact applicable
notice must be established from the imported source revision; a similarly named local
project's license is not sufficient evidence.

The `fire.jpg` style reference does not yet have documented source/license information.
Do not assume that an image of a historic artwork or a downloaded mesh has unrestricted
redistribution rights. Package source/author/license records with each external asset.

## Research acknowledgements and citation

The method builds on SeaThru/SeaThru-NeRF, 2D and 3D Gaussian Splatting, Mip-Splatting,
Radiant Foam, WaterSplatting, OceanSplat, PPISP, RawNeRF, and differentiable ray tracing.
[references.bib](references.bib) contains the corresponding entries from the paper's
bibliography. Research attribution is separate from code licensing.

The public FoamSurf citation is pending the public author list and paper URL. Use the
title **FoamSurf: A Voronoi-Surfel Representation for Differentiable Rendering**;
do not describe the submission as an accepted CVPR paper. Add the final BibTeX entry
when the public paper metadata is available.
