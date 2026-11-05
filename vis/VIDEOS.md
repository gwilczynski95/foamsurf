# Visualization clips

Every clip is produced by a committed script from a shot file in `vis/shots/`. `OUT/shot_resolved.json` next to
each video records the resolved object positions and every frame's camera.

Environment prefix (tornado, RTX 3090): see the repo's build notes; `P=foamsurf_seathru_2026-11-05`.

| Clip | Scene | Command |
|---|---|---|
| `renders/clips/curacao_spheres.mp4`: mirror and glass spheres resting on the reef, camera passing along MTN_1292–1298 (150 frames, 30 fps) | Curasao | `python vis/render_clip.py --run $P/models/Curasao-Undistorted --scene $P/data/Curasao-Undistorted --shot vis/shots/curacao_spheres.json --out renders/clips/curacao_spheres` |

Checks: `python vis/check_raytrace.py --run $P/models/Curasao-Undistorted --scene $P/data/Curasao-Undistorted --view MTN_1296`
(no objects must equal `render.py` bit-exactly).
