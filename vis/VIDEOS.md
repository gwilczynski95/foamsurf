# Visualization clips

Every clip is produced by a committed script from a shot file in `vis/shots/`. `OUT/shot_resolved.json` next to
each video records the resolved object positions and every frame's camera.

Environment prefix (tornado, RTX 3090): see the repo's build notes; `P=foamsurf_seathru_2026-11-05`.
Meshes need `pip install warp-lang` (GPU BVH) and `FALCON=<FALCON checkout>` (the Fiat 126p parts and manifest
live in `$FALCON/source/fiat_bicycle`).

| Clip | Scene | Command |
|---|---|---|
| `renders/clips/curacao_spheres.mp4`: mirror sphere floating over the reef, glass sphere just above it; camera along MTN_1296–1301 (zoom 1.1) (150 frames, 30 fps) | Curasao | `python vis/render_clip.py --run $P/models/Curasao-Undistorted --scene $P/data/Curasao-Undistorted --shot vis/shots/curacao_spheres.json --out renders/clips/curacao_spheres` |
| `renders/clips/curacao_fiat.mp4`: Fiat 126p (red paint with clear coat, FALCON's other part materials) on the sand and a mirror sphere floating next to it, same camera pass | Curasao | `python vis/render_clip.py --run $P/models/Curasao-Undistorted --scene $P/data/Curasao-Undistorted --shot vis/shots/curacao_fiat.json --out renders/clips/curacao_fiat` |
| `renders/clips/iui3_fiat.mp4`: Red Fiat 126p between a glass sphere and a floating mirror sphere on the sandy slope; 7 training views nearest MTN_5911 (zoom 1.1) | IUI3 | `python vis/render_clip.py --run $P/models/IUI3-RedSea-Undistorted --scene $P/data/IUI3-RedSea-Undistorted --shot vis/shots/iui3_fiat.json --out renders/clips/iui3_fiat` |
| `renders/clips/japanese_gardens_fiat.mp4`: Yellow Fiat on a sand patch, mirror sphere floating over the coral; lateral pass MTN_1092–1096 placed in MTN_1094 | JapaneseGradens | `python vis/render_clip.py --run $P/models/JapaneseGradens-RedSea-Undistorted --scene $P/data/JapaneseGradens-RedSea-Undistorted --shot vis/shots/japanese_gardens_fiat.json --out renders/clips/japanese_gardens_fiat` |
| `renders/clips/panama_chrome_fiat.mp4`: Chrome Fiat on the coral and a floating glass sphere; 7 views nearest MTN_1529, the poorly covered first 30 % trimmed | Panama | `python vis/render_clip.py --run $P/models/Panama-Undisorted --scene $P/data/Panama-Undisorted --shot vis/shots/panama_chrome_fiat.json --out renders/clips/panama_chrome_fiat` |
| `renders/clips/curacao_three_fiats.mp4`: Red, green and blue Fiats (FALCON's three-car garden shot) on the Curacao reef; MTN_1296–1301 | Curasao | `python vis/render_clip.py --run $P/models/Curasao-Undistorted --scene $P/data/Curasao-Undistorted --shot vis/shots/curacao_three_fiats.json --out renders/clips/curacao_three_fiats` |

Previews of a shot's anchor view (with / without shadows, reference): add `--still`.

Camera paths are chosen with `python vis/coverage.py --run ... --scene ... --shot ...`: per second of the path, the
fraction of pixels that look down at a hole or at a surface fewer than 2 training photos saw (whole frame, 10 %
border). The shots above stay under ~10 % in the border.

Checks: `python vis/check_raytrace.py --run $P/models/Curasao-Undistorted --scene $P/data/Curasao-Undistorted --view MTN_1296`
(no objects must equal `render.py` bit-exactly).
