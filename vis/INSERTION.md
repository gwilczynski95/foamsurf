# Inserting objects into a trained FoamSurf scene

How-to for the object-insertion renderer on branch `dev/raytracing` (paper: Sec. "Inserted objects and secondary
rays", supplement "Rendering inserted objects"). Every key and flag below was checked against the code.

## 1. What the branch adds

- `vis/raytrace.py`: a Whitted-style Python ray tracer that adds objects to a trained scene. The native kernel renders
  each ray up to its first object hit (a *segment*: direct light D, backscatter B, transmittance T). The object's
  light is then traced recursively with secondary rays through the same surfels and water.
- Objects: analytic spheres (mirror, glass, diffuse, metal, ceramic, gas bubbles, emissive lamps) and triangle
  meshes (`vis/mesh.py`, NVIDIA Warp BVH) with per-part materials, including textured metal/roughness (`pbr`).
- Light: matte parts are lit by the reconstructed scene plus seabed-calibrated light from above. Objects cast soft
  shadows and ambient occlusion on the surfels. Lamps (spheres or spot cones) relight the surfels and objects and
  scatter in the water.
- `vis/render_clip.py`: a still or clip renderer driven by JSON *shot* files (`vis/shots/`). Also tools to get
  models in (`export_blend.py`, `export_stl.py`, `fetch_polyhaven.py`, `specgloss_to_pbr.py`, `make_fish.py`) and
  checks (`check_raytrace.py`, `coverage.py`).
- **Visualization only.** `training/`, the models and `render.py` are unchanged (`git diff --stat slim HEAD`).
  The only native change is `rgb_forward_limited`: the forward kernel with an optional per-ray far plane
  (`ray_far`, `nullptr` = off), so training and evaluation behave exactly as on `slim`.

## 2. Setup

Use the same conda env as `slim` (Python 3.11, CUDA 12.4, torch 2.6, OptiX SDK 8.0; see `README.md`).

```bash
conda activate foamsurf
export OPTIX_DIRECTORY=/path/to/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
export TORCH_CUDA_ARCH_LIST=8.6        # 8.6 RTX 3090, 8.9 RTX 4090 (the scripts default to 8.9)
export CUDA_VISIBLE_DEVICES=0          # build.sh defaults to GPU 1
CONDA_ENV_PATH=$CONDA_PREFIX ./build.sh            # everything; renderer only: CONDA_ENV_PATH=$CONDA_PREFIX renderer/build.sh
python -c "import foamsurf._C as c; assert hasattr(c, 'rgb_forward_limited'); print('ok')"
pip install warp-lang                  # GPU BVH for meshes (tested: 1.19.0)
```

- **The extension must be rebuilt on this branch.** An extension built on `slim` lacks `rgb_forward_limited`, and
  every trace fails with an `AttributeError`. If the build picks up the wrong compiler, set
  `CC=/usr/bin/gcc CXX=/usr/bin/g++`. On the GMUM cluster, `sbatch scripts/cluster/build.sbatch` does the same.
- `plyfile` and `scipy` are already in `requirements.txt`. Clips need `ffmpeg` on `PATH`. Blender 4.x is needed
  only to *export* models (section 5).
- **Models** (`assets/models/`) are git-excluded. Extract the hand-over tarball in the repo root:
  `tar xzf foamsurf_assets_models.tar.gz`. This gives `assets/models/<name>/{manifest.json, mesh.npz, *.png}` for
  antique_ceramic_vase_01, barramundi, cannon_01, ceramic_vase_01, dutch_ship_large_01, portable_searchlight,
  treasure_chest, rubber_duck, submarine_01, submarine_toy and fish_{silver,tang,clown}.
- The Fiat 126p is not in the tarball. It comes from a FALCON checkout: `export FALCON=/path/to/FALCON` (shots use
  `$FALCON/source/fiat_bicycle/manifest.json`). See `vis/VIDEOS.md`.
- Run every command from the repo root: manifest paths in shots are relative to the working directory.

## 3. Quickstart: a glass and a mirror sphere

`RUN` is a `train.py` run directory (with `joint/` or `config.yaml` + `checkpoints/last.pt`). `DATASET` is its
COLMAP directory (pass `--scene` when the dataset has moved). `VIEW` is any training or test view name of that
dataset: an image stem such as `MTN_1296`, or `python render.py --run RUN --export-cameras cams.json` lists them.

```bash
cat > vis/shots/my_spheres.json <<'EOF'
{"anchor_view": "VIEW",
 "objects": [
  {"name": "mirror", "u": 0.35, "v": 0.75, "radius_fraction": 0.06, "height": 0.3,
   "material": "mirror", "tint": [0.95, 0.95, 0.95]},
  {"name": "glass", "u": 0.65, "v": 0.75, "radius_fraction": 0.06, "height": 0.3,
   "material": "glass", "tint": [0.98, 0.99, 1.0], "ior": 1.128}],
 "trace": {"max_depth": 4, "antialias": 8}}
EOF
python vis/render_clip.py --run RUN --scene DATASET --shot vis/shots/my_spheres.json --out renders/stills/my_spheres --still
```

Ready-made: `P=foamsurf_seathru_2026-11-05; python vis/render_clip.py --run $P/models/Curasao-Undistorted --scene
$P/data/Curasao-Undistorted --shot vis/shots/curacao_spheres.json --out renders/stills/curacao_spheres --still`.

`--still` renders only the anchor camera at its native resolution and writes three files to `OUT/`:

| file | content |
|---|---|
| `still.png` | the result, camera model (PPISP) applied, like `render.py`'s `full` |
| `still_noshadow.png` | the same with `shadows: false` (written only when shadows are on) |
| `still_reference.png` | the scene without objects (identical to `render.py` `full` of that camera) |

The console prints `still: <s> s, rays per depth [...]` and `... sphere lifted X radii out of the surface` for each
sphere that `clear` moved. **`--still` writes no `shot_resolved.json`.** That file (world centres, radii, mesh
placements and all cameras) is written only in clip mode. To get it without rendering any frame, use
`--frames 2 --only 999` without `--still` (the path is computed, but no frame index matches).

To choose `u, v`, render `{"anchor_view": "VIEW", "objects": []}` with `--still` (as `vis/shots/ruins/empty.json`
does). Then read pixel positions off `still_reference.png`: `u = x / width`, `v = y / height`.

## 4. Shot-file reference

Unknown keys in the top level and in objects are **silently ignored** (typos too). Unknown keys in `trace` raise a
`TypeError`. "World units" are the COLMAP units of the dataset, which differ per scene. In the SeaThru scenes,
objects 12–65 units from the camera are in the visible water (`vis/shots/*_showcase.json`).

### Top level

| key | default | meaning |
|---|---|---|
| `anchor_view` | required unless `camera` | dataset view (train or test) used to place objects and rendered by `--still` |
| `camera` | – | pose file (render.py format, path **relative to the shot file**). Its first camera is the anchor (any size or intrinsics) and overrides `anchor_view`. `render.py --export-cameras` gives a template |
| `objects` | required | list, processed in order (see "references" below) |
| `trace` | `{}` | `TraceSettings` overrides (table below) |
| `still_time` | 3.0 | time (s) at which bubbles are frozen for `--still` |
| `frames`, `fps` | 120, 30 | clip length (`--frames` overrides) |
| `start_time` | 3.0 | bubble time of clip frame 0 |
| `path` | `{}` | clip camera path (section 4.7) |
| `_note` | – | free text, ignored (convention: how the models were made) |

`u, v` always mean the image fraction of the **anchor camera**: pixel column `int(u·W)` from the left and row
`int(v·H)` from the top, at the anchor's native resolution. Keep `u, v < 1`: surface placement does not clamp.
"Up" is the mean up vector of the training cameras.

### 4.1 Spheres (`"type": "sphere"`, the default)

| key | default | meaning |
|---|---|---|
| `material` | `mirror` | `mirror`, `glass`, `diffuse`, `metal`, `ceramic`, `emissive` (lamp, 4.4) |
| `tint` | [1,1,1] | reflectance / transmittance / albedo |
| `ior` | 1.5/1.33 = 1.128 | glass/ceramic index **relative to water**. Below 1, a gas bubble |
| `kd` | 1.0 | metal: mirror fraction; ceramic: diffuse weight |
| `name` | none | needed to be a target of `aim` / `near` |

Placement: the first matching mode wins.

| mode | keys | centre | radius |
|---|---|---|---|
| explicit | `center` [x,y,z], `radius` | given (world) | given |
| attached to a mesh | `attach` (mesh name), `local` [x,y,z], `radius_local` (0.006) | `origin + size · local @ axes` of that mesh (4.2) | `radius_local` × mesh size |
| next to an object | `near` (name), `offset` ([0,2,0]), `radius_of_near` (0.06) | target centre + target radius × (offset[0]·right + offset[1]·up + offset[2]·towards the anchor camera), right/towards horizontal | `radius_of_near` × target radius |
| at a distance | `u`, `v`, `distance`, `radius_fraction`, `height` (0) | `distance` world units along the anchor ray (open water allowed) + `height`·R·up | `radius_fraction` × `distance` |
| in front of the surface | `u`, `v`, `depth_fraction`, `radius_fraction`, `height` (0) | at `depth_fraction` × surface distance on the ray + `height`·R·up | `radius_fraction` × surface distance |
| on the surface (default) | `u`, `v`, `radius_fraction`, `lift` (1.0), `height` (0) | surface point seen at (u,v) + (`lift`+`height`)·R·up: rests at lift 1, sinks below 1, floats with height > 0 | `radius_fraction` × camera–surface distance |

In the last three modes, `clear` (default `true`) moves the sphere up in steps of 0.1 R (at most 4 R) until no
surfel lies within 1.03 R. A sphere cutting into the reef would show the empty inside of the reconstruction.
`radius_fraction` is roughly the angular radius in radians, so the apparent size does not depend on depth.

### 4.2 Meshes (`"type": "mesh"`) and mirror panels (`"type": "mirror_panel"`)

| key | default | meaning |
|---|---|---|
| `manifest` | required (mesh) | path; `$VARS` expanded; relative to the working directory |
| `name` | `"mesh"` | give one to reference it (`attach`, `aim`, `near`, bubble `source`) |
| `materials` | – | `{part id: material}` override, merged key by key over the manifest's (4.3) |
| `hide_parts` | – | list of part ids to drop (for example a wreck's sails). Never all of them (section 8) |
| `yaw_deg` | 0 | turn about up; at 0 the mesh's +x is the anchor camera's right and +z points towards the camera; positive turns +x towards the camera |
| `pitch_deg`, `roll_deg` | 0 | pre-tilt: pitch lowers the +z end, roll lowers the +x side (foot re-set to y = 0) |
| mirror_panel only | `aspect` 0.6, `border` 0.06, `thickness` 0.04, `lean_deg` 15, `mirror`, `frame` | framed mirror, pane on +z (faces the camera at yaw 0), top leaned towards it. `mirror` / `frame` are material overrides |

Meshes are normalized: centred horizontally, lowest point at y = 0, largest extent 1. Placement modes (first match):

| mode | keys | result |
|---|---|---|
| next to an object | `near`, `offset` ([0,2,0], target radii, as for spheres), `size_of_near` (2.0) | upright, bbox centre at the near point, largest extent = `size_of_near` × target radius (meshes: bounding-sphere radius) |
| floating | `u`, `v`, `distance`, `size` | upright, bbox centre `distance` units along the anchor ray, largest extent `size` units |
| on the seabed (default) | `u`, `v`, `span_fraction`, `sink` (0) | foot on the surface seen at (u,v), largest extent = `span_fraction` × hit distance, tilted to a plane fitted under its footprint (at most 10°), lowered by `sink` × size. Negative sink floats it (`panama_treasure`: duck at −6) |

The placement frame is stored as `origin` (local (0,0,0), the foot centre), `size` and `axes` (rows: local x, y, z
in world). Attached points are `origin + size · local @ axes`. Read `local` coordinates off the normalized mesh:
`python -c "from vis.mesh import load_manifest; v=load_manifest('assets/models/submarine_01/manifest.json')[0]; print(v.min(0), v.max(0))"`.

### 4.3 Materials

Spheres use `material`/`tint`/`ior`/`kd`. Mesh parts (manifest and `materials` overrides) use FALCON's names:
`mode`, `rgb`, `eta`, `kd`, `maps`. Missing fields default to `ceramic`, [1,1,1], **eta 1.5**, kd 1.0.

| mode | behaviour |
|---|---|
| `mirror` | reflection × tint |
| `glass` | Schlick Fresnel split. The refracted ray crosses the solid (no water inside) and leaves at the far side (sphere: analytic; mesh: next hit on the same mesh; an open mesh acts as a thin sheet). Total internal reflection follows the reflected ray. Tint is applied once. `ior`/`eta` is relative to water: real glass is 1.128, and the mesh default 1.5 is denser |
| `diffuse` | albedo `tint` × matte light: the scene radiance at the object's centre (`environment_samples` directions), plus light from above calibrated on the seabed, × ambient occlusion by meshes, + lamps |
| `metal` | tint × (kd · mirror + (1−kd) · matte) |
| `ceramic` | Fresnel(ior) mirror coat + (1−F) · kd · tint · matte (car paint, glazed pottery) |
| `pbr` | `maps`: `albedo` (sRGB), `metal`, `rough` (greyscale), paths relative to the manifest, sampled at the mesh UVs. `rgb` multiplies the albedo (`panama_treasure` brightens the chest with [2.5,2.5,2.5]). Missing maps: metal 0, rough 0.5. One mirror ray weighted by F·(1−rough)², the rest matte, no glossy blur |
| `emissive` | spheres only (a lamp, 4.4). Not supported on mesh parts |

### 4.4 Lamps (spheres with `"material": "emissive"`)

A lamp is placed like any sphere (all six modes). It casts no shadow and does not block other lamps.

| key | default | meaning |
|---|---|---|
| `power` | 10 | intensity: irradiance at 1 world unit, without water, = `power` × the scene's light on an upward-facing surface. Falls off as e^(−c r)/r² in world units, so it depends on scene scale (Panama spots 250–300, ruins teaser 700–2500) |
| `tint` | [1,1,1] | colour (bulb radiance = tint · power / (π R²), saturates) |
| spot axis, first match | – | `aim_local` [x,y,z] in the attached mesh's frame (needs `attach`) · `aim_uv` [u,v]: the surface seen there in the anchor view · `aim`: centre of a named earlier object · `spot` [x,y,z] world axis · `down_deg` (+ `yaw_deg`): straight down tilted by down_deg towards the anchor camera's right turned by yaw. None: omnidirectional |
| `cone_deg` / `softness_deg` | 20/5 (aim_local, aim_uv, aim), 30/8 (spot, down_deg) | full intensity inside cone−softness, zero beyond cone+softness, smoothstep between |

Example (from `vis/shots/ruins/teaser.json`): a headlight on the submarine aimed at a seabed point:
`{"attach": "submarine", "local": [0.507, 0.2255, 0.031], "radius_local": 0.006, "material": "emissive",
"tint": [1.0, 0.98, 0.93], "power": 1300, "aim_uv": [0.46, 0.8], "cone_deg": 12, "softness_deg": 4}`.

### 4.5 Fish schools (`"type": "school"`)

`count` copies of one small mesh merged into one object (`build_school`). Required: `manifest`, `u`, `v`,
`distance` (centre on the anchor ray). Optional: `name` ("school"), `materials`, `forward` ("x"; the mesh axis the
head points along: x, -x, z, -z; the barramundi uses "z"), `yaw_deg` (heading, 0 = anchor right, about up),
`count` 20, `length` 0.25 (world units per fish), `spread` [2.0, 0.6, 1.0] (ellipsoid half-axes, world units:
along the heading, up, across), `gap` 1.1 (min. spacing in lengths), `size_jitter` 0.15, `heading_jitter_deg` 10,
`pitch_jitter_deg` 6, `seed` 0. No `hide_parts`/pitch/roll. A school can be a `near`/`aim` target, but not an
`attach`/bubble source.

### 4.6 Bubbles (`"type": "bubbles"`)

A stream of gas spheres (glass, `ior` 1/1.333, no shadow) from a point of a named mesh, deterministic in time.
Required: `source` (mesh/mirror_panel name), `local` [x,y,z]. Optional, lengths in mesh sizes: `direction`
[0,0,1] (local frame), `rate` 14 /s, `life` 4 s, `rise` 0.6 /s, `speed` 0.5 /s, `settle` 0.4 s, `radius`
[0.012, 0.03], `grow` 0.1 /s, `wobble` 0.03, `wobble_hz` 1.5, `spread` 0.01, `drift` 0.06 /s, `seed` 0, `tint`,
`ior`. Stills use `still_time`.

**References:** `attach`, `aim` and `near` look only at objects listed earlier (`next(...)`, the first one with that
name). A bubble `source` can be anywhere. `attach` and `source` need a mesh or mirror panel with an explicit `name`.

### 4.7 Trace settings (`"trace": {...}`, `vis/raytrace.py: TraceSettings`)

| key | default | meaning |
|---|---|---|
| `max_depth` | 4 | bounces; past it an object contributes black |
| `min_weight` | 1e-3 | secondary rays carrying less (max channel) are dropped |
| `epsilon` | 1e-4 | secondary ray offset, in object radii |
| `antialias` | 4 | extra rays per pixel on object outlines (dilated 1 px) and on mirror/metal/glass spheres; ≤ 4 fixed pattern, more = Fibonacci lattice; 0 off. Shots use 8 |
| `ambient` | [0,0,0] | extra constant light on matte surfaces (raw units) |
| `water_light_max_up`, `water_light_max_gain` | 0.6, 2.0 | limit of the extrapolated water light for rays looking up (mirrors). `null` = trained, unlimited |
| `shadows` | true | object shadows on the surfels (only D is darkened) |
| `shadow_strength`, `shadow_cone_deg` | 0.6, 25 | share and half-angle of the light cone around up |
| `occlusion_strength` | 1.0 | ambient-occlusion part of the shadow |
| `shadow_samples` | 96 | mesh shadow rays per point (cone and occlusion) |
| `shadow_blur_px` | 3.0 | image-space smoothing of the shadow factor (0 off) |
| `environment_samples` | 256 | directions for the matte light at an object's centre |
| `occlusion_samples`, `occlusion_distance` | 16, 0.5 | AO rays per matte hit and their reach (object radii) |
| `seabed_albedo`, `sky_exponent` | 0.7, 2.0 | light from above calibrated on the seabed; `null` = environment only |
| `lamps` | true | lamp lighting and in-scattering (false: bulbs still visible, they emit no light) |
| `lamp_g` | 0.9 | Henyey–Greenstein g of the water for lamp light. **Use 0.75 for beams seen from the side** |
| `lamp_samples` | 24 | equiangular samples per ray and lamp |
| `lamp_volume_shadows` | true | camera-ray samples test lamp visibility (beams, shadows in the water) |
| `lamp_blur_px` | 2.5 | image-space smoothing of lamp light on surfels (glow: ×1.5) |

### 4.8 Clips (`path` and flags)

Without `--still`, the camera follows a Catmull–Rom spline through training camera positions, and frames go to
`OUT/frames/NNNN.png`, then `OUT.mp4` (ffmpeg) and `OUT/shot_resolved.json`. The `path` keys are
`look_at_objects` (0.7; 0 = the photos' own direction, 1 = aim at the objects' mean centre), `views` (names to
pass through, in order), `nearest` (N training views nearest the anchor; otherwise all training views, sorted
along their main axis), `scale` (1.0, image size), `zoom` (1.0, focal factor) and `trim` ([0,1], part of the
spline). Flags: `--frames N`, `--only I [I ...]` (just these frames, no video), `--part K N` (contiguous chunk K
of N, no video; cluster: `scripts/cluster/render_clip.sbatch`, frames assembled afterwards). Exposure is fixed
at the anchor's camera model.

## 5. Adding a new model

All exporters write `assets/models/<name>/manifest.json` + `mesh.npz` (+ textures):
`{"format": "npz", "mesh": "mesh.npz", "parts": [{"id": ..., "material": {...}}]}`. The npz holds `vertices` [V,3]
(y-up), `faces` [F,3], `part` [F] (index into `parts`) and `corner_uv` [F,3,2] (required, zeros if untextured).
The older FALCON format, `{"parts": [{"id", "ply", "material"}], "flip_yz": true}`, is also read: PLYs are taken
as y-down unless `"flip_yz": false`, and there are no UVs, so no `pbr` textures.

- **Poly Haven (CC0):** `python vis/fetch_polyhaven.py ASSET [ASSET ...] [--res 1k] [--size 1024] [--blender /path/to/blender]`
  downloads the glTF (the .blend files are Blender 5 format) to `assets/models/raw/ASSET/` and exports to
  `assets/models/ASSET/`. The tarball's models came from
  `python vis/fetch_polyhaven.py antique_ceramic_vase_01 cannon_01 ceramic_vase_01 dutch_ship_large_01 portable_searchlight`.
  `treasure_chest` and `rubber_duck` came from the older .blend route below (asset `rubber_duck_toy`, renamed).
- **Any glTF/glb:** `blender -b --python vis/export_blend.py -- assets/models/NAME --import MODEL.glb [--size 2048]`.
  The barramundi is Khronos glTF-Sample-Assets `BarramundiFish.glb` (CC0):
  `blender -b --python vis/export_blend.py -- assets/models/barramundi --import assets/models/raw/barramundi/BarramundiFish.glb`.
- **Any .blend (Blender 4.x readable):** `blender -b MODEL.blend --python vis/export_blend.py -- assets/models/NAME`.
  All render-visible mesh objects are exported. Each material slot becomes one part, with Principled BSDF Base
  Color / Metallic / Roughness image textures (packed glTF channels handled). A material **without image textures
  exports as white `pbr`**, so override it in the shot.
- **Spec/gloss textures:** `python vis/specgloss_to_pbr.py DIFFUSE SPECULAR GLOSS OUT_DIR PREFIX --manifest OUT_DIR/manifest.json`
  (the toy submarine, see the `_note` of `vis/shots/iui3_toy_submarine.json`).
- **Untextured STL (binary):** `python vis/export_stl.py MODEL.stl assets/models/NAME [--hull_size 0.03] [--pipe_ratio 4] [--stripes '[[x0,x1],...]'] [--paint paint.json]`
  splits the model by connected components into parts `hull`, `stripe`, `pipe`, `detail` with default paints.
  submarine_01: `python vis/export_stl.py submarine-01.stl assets/models/submarine_01 --hull_size 0.02 --stripes '[[-22,-15],[-6,1],[10,17]]'`.
- **Procedural fish:** `python vis/make_fish.py assets/models/fish_silver --kind silver` (`tang`, `clown`).

List the part ids with `python -c "import json,sys; print([p['id'] for p in json.load(open(sys.argv[1]))['parts']])" assets/models/NAME/manifest.json`.
Override parts in the shot, field by field:
`"materials": {"caroseria": {"mode": "ceramic", "rgb": [0.6, 0.05, 0.035], "kd": 1.0, "eta": 2.2}}`.
`{"rgb": [2.5, 2.5, 2.5]}` keeps a pbr part's maps and scales its albedo.

## 6. Checking correctness

1. **Bit-exact without objects:** `python vis/check_raytrace.py --run RUN --scene DATASET [--view TEST_VIEW] [--out renders/raytrace_check]`
   (`--view` must be a *test* view; default: the first). It prints `max |raw - render.py raw|` and
   `max |full - render.py full|`, which must both be `0.000e+00`, and the same for a sphere far behind every
   surface. It also writes `<view>_reference.png`, `<view>_spheres.png` and `<view>_spheres_noshadow.png`
   (mirror at (0.40, 0.72), glass at (0.64, 0.68)).
2. **Every still:** `still_reference.png` is the plain model. Away from objects, `still.png` must match it except
   for shadows, lamp light and glow. `still_noshadow.png` isolates the shadows. Glass spheres show the background
   inverted, and mirrors show the scene and the water above.
3. **Placement in world units:** `shot_resolved.json` (clip mode, or the `--frames 2 --only 999` trick) has every
   sphere's `center`/`radius` and every mesh's `placement` (`anchor`, `size`, `origin`, `axes`, and for seabed
   meshes `depth` and `footprint_hits` out of 49).
4. **Camera paths (clips):** `python vis/coverage.py --run RUN --scene DATASET --shot SHOT [--scale 0.25]` prints,
   per second, the fraction of pixels that look down at a hole or at a surface fewer than 2 photos saw (whole
   frame, 10 % border). Keep the border under ~10 % (`trim`, `nearest`, `zoom`). It needs `anchor_view` (not
   `camera`) and at least one object.

## 7. Recipes for reviewer requests

**Attenuation with distance:** identical spheres (world radius R) along rays at growing distance, with
`radius_fraction = R / distance`, as in `vis/shots/*_showcase.json` (Curacao R = 0.68):

```json
{"u": 0.86, "v": 0.62, "distance": 12, "radius_fraction": 0.05674, "material": "diffuse", "tint": [0.85, 0.85, 0.85]},
{"u": 0.83, "v": 0.45, "distance": 20, "radius_fraction": 0.03405, "material": "diffuse", "tint": [0.85, 0.85, 0.85]},
{"u": 0.81, "v": 0.32, "distance": 30, "radius_fraction": 0.0227,  "material": "diffuse", "tint": [0.85, 0.85, 0.85]}
```

Use red `tint` to show colour loss (`panama_showcase_red.json`). For a mesh, use `"distance": D, "size": S` with
constant S. For a few renders of one object at several distances, make one shot per distance and keep `u, v`.

**Glass vs mirror vs diffuse, same object:** for spheres, change `material` only. For a mesh, override every part:

```bash
python - <<'EOF'
import copy, json
base = json.load(open("vis/shots/my_shot.json"))            # objects[0] is the mesh
parts = [p["id"] for p in json.load(open("assets/models/ceramic_vase_01/manifest.json"))["parts"]]
for name, m in {"glass": {"mode": "glass", "rgb": [0.98, 0.99, 1.0], "eta": 1.128},
                "mirror": {"mode": "mirror", "rgb": [0.95, 0.95, 0.95]},
                "diffuse": {"mode": "diffuse", "rgb": [0.8, 0.8, 0.8]}}.items():
    shot = copy.deepcopy(base); shot["objects"][0]["materials"] = {p: m for p in parts}
    json.dump(shot, open(f"vis/shots/my_shot_{name}.json", "w"), indent=1)
EOF
```

**Lamp on/off:** same shot twice. `"trace": {..., "lamps": false}` keeps the geometry identical but bulbs still
show. For a clean "off", drop the emissive objects. `panama_showcase.json` vs `panama_lamp.json` (omni lamp,
power 20, 1.5 radii above the reef) and `panama_spot.json` (spot aimed at the Fiat) are existing pairs.

**The same insertion in all four SeaThru scenes:** `u, v` refer to one view, so write one shot per scene. Anchor
views used so far: Curasao `MTN_1296`, IUI3 `MTN_5911`, JapaneseGradens `MTN_1094`, Panama `MTN_1529`. Scene
scales differ (showcase sphere radius 0.68 / 0.56 / 0.39 / 0.46 units). For a comparable apparent size, use
relative sizes (`radius_fraction`, `span_fraction` on the surface) rather than `distance`/`size`.

```bash
P=foamsurf_seathru_2026-11-05   # data; water-anchor models: M=water_anchor_2026-11-07/models
for s in curacao:Curasao-Undistorted iui3:IUI3-RedSea-Undistorted \
         japanese_gardens:JapaneseGradens-RedSea-Undistorted panama:Panama-Undisorted; do
  python vis/render_clip.py --run $P/models/${s#*:} --scene $P/data/${s#*:} \
      --shot vis/shots/${s%%:*}_showcase.json --out renders/stills/${s%%:*}_showcase --still
done
```

**Higher-resolution figure:** put a pose file under `"camera"` (export with `render.py --export-cameras`, keep
one record, scale `width`, `height`, `fx`, `fy`, `cx`, `cy`), as `vis/shots/ruins/teaser_camera.json` does
(2160×800; `teaser_camera_half.json` is the fast preview). The `ruins` shots belong to the synthetic ruins scene
(C16 V1), not to SeaThru. With `foamsurf_teaser_ruiny_C16_v1.zip` extracted to `scenes/` (git-excluded):

```bash
P=scenes/foamsurf_teaser_ruiny_C16_v1
python vis/render_clip.py --run $P/model --scene $P/dataset/ruins_C16_800_C_20261112 \
    --shot vis/shots/ruins/teaser.json --out renders/teaser/final_v2 --still   # ~245 s on an RTX 3090
```

## 8. Pitfalls and performance

- **`hide_parts` listing every part** leaves an empty mesh, which ends in a CUDA *illegal memory access*.
- **Images larger than the training images:** the ray buffer is sized from the training images. For a `camera`
  pose file, `render_clip` raises it before loading the run (`_raise_ray_buffer` patches `rays_per_batch` to
  width × height). Clips from `anchor_view` with `path.scale > 1` do not get this.
- **OOM on close-ups with meshes:** shadow and occlusion rays are batched (`RAY_CHUNK = 4_000_000` in
  `vis/raytrace.py`; lower it if needed). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` also helps.
- **Lamp beams** are visible only near their source in clear water (1/r² dominates the in-scattering). With the
  default `lamp_g` 0.9 (forward-peaked), beams seen from the side vanish; 0.75 works (ruins teaser).
- `u` or `v` equal to 1.0 fails with an index error for surface placement. `clear` may lift spheres placed at
  the surface (the console reports it).
- A mesh's attached points (`local`, `aim_local`, bubbles) use the placement frame. A `pitch_deg`/`roll_deg`
  tilt of the vertices is not applied to them.
- Glass mesh parts default to `eta` 1.5 relative to water. Matte light is evaluated once at each object's centre,
  so a very large object (the wreck) gets uniform ambient light.
- Shared GPU: set `CUDA_VISIBLE_DEVICES`. The cluster (`rtx4090_batch`) is ~3× faster for batches.
- **Times (RTX 3090):** a 2160×800 still with ~10 objects, 6 lamps and `antialias` 8 took ~245 s. Panama clip
  frames with bubbles took ~80 s each. Iterate at half resolution and `antialias` 0–4, then render the final
  version once.
