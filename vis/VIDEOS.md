# Visualization clips

Every clip is produced by a committed script from a shot file in `vis/shots/`. `OUT/shot_resolved.json` next to
each video records the resolved object positions and every frame's camera.

Environment prefix (tornado, RTX 3090): see the repo's build notes; `P=foamsurf_seathru_2026-11-05`.
Meshes need `pip install warp-lang` (GPU BVH) and `FALCON=<FALCON checkout>` (the Fiat 126p parts and manifest
live in `$FALCON/source/fiat_bicycle`).

| Clip | Scene | Command |
|---|---|---|
| `renders/clips/curacao_spheres.mp4`: mirror and glass spheres resting on the reef, camera passing along MTN_1292–1298 (150 frames, 30 fps) | Curasao | `python vis/render_clip.py --run $P/models/Curasao-Undistorted --scene $P/data/Curasao-Undistorted --shot vis/shots/curacao_spheres.json --out renders/clips/curacao_spheres` |
| `renders/clips/curacao_fiat.mp4`: Fiat 126p (red paint with clear coat, FALCON's other part materials) on the sand and a mirror sphere reflecting it, same camera pass | Curasao | `python vis/render_clip.py --run $P/models/Curasao-Undistorted --scene $P/data/Curasao-Undistorted --shot vis/shots/curacao_fiat.json --out renders/clips/curacao_fiat` |

Previews of a shot's anchor view (with / without shadows, reference): add `--still`.

Checks: `python vis/check_raytrace.py --run $P/models/Curasao-Undistorted --scene $P/data/Curasao-Undistorted --view MTN_1296`
(no objects must equal `render.py` bit-exactly).
