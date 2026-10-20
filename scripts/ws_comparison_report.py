#!/usr/bin/env python3
"""FoamSurf vs WaterSplatting on the SeaThru test views, image by image (HTML).

Usage: python scripts/ws_comparison_report.py --ours outputs/seathru/<job> --ws <water-splatting run>
       --output report.html [--seed 42] [--iteration last|best]

Both methods are scored from their saved 8-bit test renders against the dataset images:
PSNR on the whole frame, on the seabed and on the water (SAM masks, configs/water_sam).
WaterSplatting renders its final checkpoint, so FoamSurf uses its last evaluation by
default (--iteration best picks the best test iteration instead, which favours FoamSurf).
Per view: GT / WaterSplatting / FoamSurf, both error maps, the map of where each method
is better, crops of the blocks FoamSurf loses most, and both decompositions.
"""
import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam_water_mask import PROMPTS, load_masks  # noqa: E402
from seabed_focus_report import thumbnail  # noqa: E402
from seathru_report import SHORT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASETS = Path("/shared/sets/datasets/SeathruNeRF_dataset")
BLOCK = 96  # crop size in pixels
WORST = 3
ERROR_MAX = 0.15  # error maps saturate here
DIFF_MAX = 0.08
OURS_BETTER, WS_BETTER, NEUTRAL = np.array([42, 120, 214]), np.array([235, 104, 52]), np.array([150, 150, 150])


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def psnr(prediction, target, mask=None):
    error = (prediction - target) ** 2
    if mask is not None:
        if not mask.any():
            return float("nan")
        error = error[mask]
    return float(-10 * np.log10(max(error.mean(), 1e-12)))


def error_map(prediction, target):
    """Mean absolute error as an inferno-like heat map."""
    from matplotlib import colormaps
    error = np.abs(prediction - target).mean(-1) / ERROR_MAX
    return Image.fromarray((colormaps["inferno"](np.clip(error, 0, 1))[..., :3] * 255).astype(np.uint8))


def difference_map(ours, ws, target):
    """Blue where FoamSurf is closer to the target, orange where WaterSplatting is, gray when equal."""
    delta = np.clip((np.abs(ws - target).mean(-1) - np.abs(ours - target).mean(-1)) / DIFF_MAX, -1, 1)[..., None]
    picture = np.where(delta > 0, NEUTRAL + delta * (OURS_BETTER - NEUTRAL), NEUTRAL - delta * (WS_BETTER - NEUTRAL))
    return Image.fromarray(picture.astype(np.uint8))


def worst_blocks(ours, ws, target, count=WORST, size=BLOCK):
    """Top-left corners of the non-overlapping blocks where FoamSurf's MSE exceeds WaterSplatting's most."""
    height, width = target.shape[:2]
    rows, columns = height // size, width // size
    excess = (((ours - target) ** 2).mean(-1) - ((ws - target) ** 2).mean(-1))[:rows * size, :columns * size]
    per_block = excess.reshape(rows, size, columns, size).mean((1, 3))
    order = np.argsort(per_block.ravel())[::-1][:count]
    return [(int(i // columns) * size, int(i % columns) * size, float(per_block.ravel()[i])) for i in order]


def crop(image, top, left, size=BLOCK, scale=2):
    patch = Image.fromarray((image[top:top + size, left:left + size] * 255).round().astype(np.uint8))
    return patch.resize((size * scale, size * scale), Image.NEAREST)


def best_or_last(run, which):
    rows = [r for r in csv.DictReader((run / "joint/metrics/eval.csv").open()) if r.get("eval/hybrid_psnr")]
    if which == "best":
        return int(max(rows, key=lambda r: float(r["eval/hybrid_psnr"]))["iteration"])
    return max(int(float(r["iteration"])) for r in rows)


def fmt(value, digits=2, signed=False):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return (f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}").replace(".", ",").replace("-", "−")


def delta_cell(value):
    cls = "" if math.isnan(value) else (" up" if value > 0.05 else " down" if value < -0.05 else "")
    return f"<td class='num{cls}'>{fmt(value, 2, signed=True)}</td>"


def compare_scene(scene, ours_run, ws_run, iteration, masks):
    folder = ours_run / "joint/evaluation" / f"iteration_{iteration:06d}"
    views = []
    for index, view in enumerate(PROMPTS[scene]):
        target = load_rgb(DATASETS / scene / "images" / f"{view}.png")
        ours = load_rgb(folder / f"{index:04d}_{view}_hybrid.png")
        ws = load_rgb(ws_run / "renders/rgb" / f"{view}.png")
        water = masks[view]
        scores = {}
        for name, image in (("ours", ours), ("ws", ws)):
            scores[name] = dict(all=psnr(image, target), seabed=psnr(image, target, ~water),
                                water=psnr(image, target, water))
        views.append(dict(view=view, index=index, target=target, ours=ours, ws=ws, scores=scores, water=water))
    return views


def view_html(scene, view, ours_run, ws_run, iteration):
    s = view["scores"]
    header = (f"<h3>{view['view']}</h3><p class='small'>PSNR cały kadr: WS {fmt(s['ws']['all'])} / my "
              f"{fmt(s['ours']['all'])} · dno: WS {fmt(s['ws']['seabed'])} / my {fmt(s['ours']['seabed'])} · woda: WS "
              f"{fmt(s['ws']['water'])} / my {fmt(s['ours']['water'])} (woda {100 * view['water'].mean():.0f}% kadru)</p>")
    main = [("GT", Image.fromarray((view["target"] * 255).astype(np.uint8))),
            ("WaterSplatting", Image.fromarray((view["ws"] * 255).round().astype(np.uint8))),
            ("FoamSurf (my)", Image.fromarray((view["ours"] * 255).round().astype(np.uint8))),
            ("błąd WaterSplatting", error_map(view["ws"], view["target"])),
            ("błąd FoamSurf", error_map(view["ours"], view["target"])),
            ("kto lepszy: niebieski = my, pomarańczowy = WS", difference_map(view["ours"], view["ws"], view["target"]))]
    grid = "".join(f"<figure><img loading='lazy' src='{thumbnail(image, 520)}' alt=''><figcaption>{label}</figcaption>"
                   f"</figure>" for label, image in main)
    crops = []
    for top, left, excess in worst_blocks(view["ours"], view["ws"], view["target"]):
        trio = "".join(f"<figure><img loading='lazy' src='{thumbnail(crop(image, top, left), 2 * BLOCK)}' alt=''>"
                       f"<figcaption>{label}</figcaption></figure>"
                       for label, image in (("GT", view["target"]), ("WS", view["ws"]), ("my", view["ours"])))
        crops.append(f"<div class='crop'><div class='small'>fragment x={left}, y={top} (Δ MSE {excess:.4f})</div>"
                     f"<div class='trio'>{trio}</div></div>")
    folder = ours_run / "joint/evaluation" / f"iteration_{iteration:06d}"
    parts = [("WS: obiekt bez wody (rgb_clear)", ws_run / "renders/rgb_clear" / f"{view['view']}.png"),
             ("WS: medium (rgb_medium)", ws_run / "renders/rgb_medium" / f"{view['view']}.png"),
             ("my: Gaussy na czarnym", folder / f"{view['index']:04d}_{view['view']}_gaussian_only_black.png"),
             ("my: backscatter medium", folder / f"{view['index']:04d}_{view['view']}_backscatter.png")]
    decomposition = "".join(f"<figure><img loading='lazy' src='{thumbnail(Image.open(path), 380)}' alt=''>"
                            f"<figcaption>{label}</figcaption></figure>" for label, path in parts if path.is_file())
    return (f"{header}<div class='grid'>{grid}</div><h4>Fragmenty, na których przegrywamy najbardziej (2× powiększenie)</h4>"
            f"<div class='crops'>{''.join(crops)}</div><h4>Rozkład na składowe</h4><div class='parts'>{decomposition}</div>")


def summary_table(results):
    rows = []
    means = {key: [] for key in ("ws_all", "ours_all", "ws_seabed", "ours_seabed", "ws_water", "ours_water")}
    for scene, views in results:
        values = {}
        for method in ("ws", "ours"):
            for region in ("all", "seabed", "water"):
                values[f"{method}_{region}"] = float(np.nanmean([v["scores"][method][region] for v in views]))
                means[f"{method}_{region}"].append(values[f"{method}_{region}"])
        rows.append((SHORT[scene], values))
    rows.append(("<b>średnia ze scen</b>", {key: float(np.nanmean(v)) for key, v in means.items()}))
    body = "".join(
        f"<tr><td>{label}</td>"
        + "".join(f"<td class='num'>{fmt(v[f'ws_{r}'])}</td><td class='num'>{fmt(v[f'ours_{r}'])}</td>"
                  + delta_cell(v[f"ours_{r}"] - v[f"ws_{r}"]) for r in ("all", "seabed", "water"))
        + "</tr>" for label, v in rows)
    head = "".join(f"<th colspan='3'>{title}</th>" for title in ("PSNR cały kadr", "PSNR dno", "PSNR woda"))
    sub = "<th class='num'>WS</th><th class='num'>my</th><th class='num'>my − WS</th>" * 3
    return f"<div class='tw'><table><tr><th rowspan='2'>scena</th>{head}</tr><tr>{sub}</tr>{body}</table></div>"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ours", type=Path, required=True, help="train_seathru.sbatch output (outputs/seathru/<job>)")
    parser.add_argument("--ws", type=Path, required=True, help="WaterSplatting run (outputs/seathru/<job>)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", default="", help="run name suffix after the scene, e.g. _sh3")
    parser.add_argument("--iteration", choices=("last", "best"), default="last")
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    masks = load_masks(ROOT / "configs/water_sam")
    results, panels = [], []
    for scene in PROMPTS:
        name = scene + args.variant + ("" if args.seed == 42 else f"_seed{args.seed}")
        ours_run, ws_run = args.ours / name, args.ws / scene
        if not (ours_run / "joint/metrics/eval.csv").is_file() or not (ws_run / "renders/rgb").is_dir():
            continue
        iteration = best_or_last(ours_run, args.iteration)
        views = compare_scene(scene, ours_run, ws_run, iteration, masks)
        results.append((scene, views))
        ws_eval = json.loads((ws_run / "renders/render_eval.json").read_text())
        body = (f"<p class='small'>FoamSurf: {html.escape(name)}, iteracja {iteration}; WaterSplatting: krok "
                f"{ws_eval.get('step')}, {ws_eval.get('num_gaussians')} Gaussów.</p>"
                + "".join(view_html(scene, view, ours_run, ws_run, iteration) for view in views))
        panels.append((SHORT[scene], body))
    tabs = "".join(f'<button class="tab" data-tab="{i}" aria-pressed="{str(i == 0).lower()}">{html.escape(l)}</button>'
                   for i, (l, _) in enumerate(panels))
    bodies = "".join(f'<div class="panel" data-tab="{i}"{"" if i == 0 else " hidden"}>{b}</div>'
                     for i, (_, b) in enumerate(panels))
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in args.note)
    which = "ostatniej ewaluacji (jak WaterSplatting: końcowy checkpoint)" if args.iteration == "last" else \
        "najlepszej iteracji na zbiorze testowym (na korzyść FoamSurf)"
    page = TEMPLATE.replace("{{BODY}}", f"""
<h1>FoamSurf vs WaterSplatting: widoki testowe SeaThru-NeRF</h1>
<p class="lead">Oba modele oceniane z zapisanych renderów 8-bit względem obrazów datasetu, ten sam podział
(co 8. obraz według nazwy), pełna rozdzielczość. FoamSurf z {which}, seed {args.seed}. Dno i woda: maski SAM.</p>
{summary_table(results)}
<ul class="small">{notes}</ul>
<div class="tabs">{tabs}</div>{bodies}""")
    args.output.write_text(page)
    print(args.output)


TEMPLATE = """<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FoamSurf vs WaterSplatting</title>
<style>
:root{--bg:#fcfcfb;--panel:#fff;--text:#0b0b0b;--muted:#52514e;--line:#e4e3df;--up:#0f7a4f;--down:#b3402a;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--panel:#222220;--text:#f0efec;
--muted:#c3c2b7;--line:#3a3936;--up:#4cc38a;--down:#f08060;--accent:#3987e5}}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#222220;--text:#f0efec;--muted:#c3c2b7;--line:#3a3936;--up:#4cc38a;
--down:#f08060;--accent:#3987e5}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}
main{max-width:1640px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.55rem;margin:0 0 6px} h3{font-size:1.1rem;margin:30px 0 4px;border-bottom:1px solid var(--line)}
h4{font-size:.95rem;margin:14px 0 6px} .small{font-size:.8rem;color:var(--muted)} .lead{color:var(--muted);max-width:980px}
.tw{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:.88rem;margin:10px 0}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums} td.up{color:var(--up);font-weight:600} td.down{color:var(--down);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:8px}
.parts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:8px}
figure{margin:0} figure img{width:100%;border-radius:4px;display:block;image-rendering:auto}
figcaption{font-size:.8rem;color:var(--muted);margin-top:2px}
.crops{display:flex;flex-wrap:wrap;gap:14px} .crop{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px}
.trio{display:flex;gap:6px} .trio figure{width:180px} .trio img{image-rendering:pixelated}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0}
.tab{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:6px;padding:4px 12px;cursor:pointer}
.tab[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);font-weight:600}
</style></head><body><main>{{BODY}}</main>
<script>
document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(t => t.setAttribute('aria-pressed', String(t === tab)));
  document.querySelectorAll('.panel').forEach(p => p.hidden = p.dataset.tab !== tab.dataset.tab);
}));
</script></body></html>
"""


if __name__ == "__main__":
    main()
