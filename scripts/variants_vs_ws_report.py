#!/usr/bin/env python3
"""Variants of a train_seathru.sbatch sweep against a control run and WaterSplatting (HTML).

Usage: python scripts/variants_vs_ws_report.py --sweep outputs/seathru/<job> --control outputs/seathru/<job>
       --ws <water-splatting run> --output report.html [--seed 42] [--note ...]

Every number comes from the saved 8-bit test renders at the LAST evaluation (WaterSplatting
renders its final checkpoint): PSNR on the frame, the seabed and the water (SAM masks), and
the MSE split into low frequencies (blur sigma 16 px: colour, brightness, haze) and the
rest (detail). Also FPS (median of the last 10 evaluations) and the Gaussian count.
Per scene: a table, the gain of every variant over the control, and GT / WaterSplatting /
control / best variant with the map of where the variant beats WaterSplatting.
"""
import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_comparison_report as w  # noqa: E402
from sam_water_mask import PROMPTS, load_masks  # noqa: E402
from seabed_focus_report import thumbnail  # noqa: E402
from seathru_report import SHORT, chart, fps_median  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASETS = Path("/shared/sets/datasets/SeathruNeRF_dataset")
LOW_SIGMA = 16


def frequency_split(prediction, target):
    error = prediction - target
    low = np.stack([gaussian_filter(error[..., c], LOW_SIGMA) for c in range(3)], -1)
    return float((low ** 2).mean()), float(((error - low) ** 2).mean())


def score_views(scene, images, masks):
    """images: view -> prediction array; mean metrics over the scene's test views."""
    rows = []
    for view in PROMPTS[scene]:
        target = w.load_rgb(DATASETS / scene / "images" / f"{view}.png")
        prediction, water = images[view], masks[view]
        low, high = frequency_split(prediction, target)
        rows.append(dict(all=w.psnr(prediction, target), seabed=w.psnr(prediction, target, ~water),
                         water=w.psnr(prediction, target, water), low=low, high=high))
    return {key: float(np.nanmean([r[key] for r in rows])) for key in rows[0]}


def ours_images(run, iteration):
    folder = run / "joint/evaluation" / f"iteration_{iteration:06d}"
    scene = next(s for s in PROMPTS if run.name.startswith(s))
    return {view: w.load_rgb(folder / f"{i:04d}_{view}_hybrid.png") for i, view in enumerate(PROMPTS[scene])}


def gaussians(run):
    summary = run / "joint/summary.json"
    return json.loads(summary.read_text())["gaussian_count"] / 1000 if summary.is_file() else float("nan")


def collect(run, masks):
    scene = next(s for s in PROMPTS if run.name.startswith(s))
    iteration = w.best_or_last(run, "last")
    values = score_views(scene, ours_images(run, iteration), masks)
    values.update(fps=fps_median(run), gaussians=gaussians(run), iteration=iteration)
    return values


COLUMNS = (("all", "PSNR", 2), ("seabed", "PSNR dna", 2), ("water", "PSNR wody", 2),
           ("low", "MSE kolor/mgła ×1e-4", 1), ("high", "MSE detal ×1e-4", 1),
           ("fps", "FPS", 0), ("gaussians", "Gaussy [tys.]", 0))


def scene_table(rows, control, ws):
    head = "".join(f"<th class='num'>{html.escape(label)}</th>" for _, label, _ in COLUMNS)
    body = []
    for label, values, emphasis in rows:
        cells = []
        for key, _, digits in COLUMNS:
            value = values.get(key, float("nan"))
            shown = value * 1e4 if key in ("low", "high") else value
            extra = ""
            if key in ("all", "seabed", "water") and label not in ("WaterSplatting",):
                gap = value - ws[key]
                extra = f"<br><span class='small {'up' if gap > 0 else 'down'}'>{w.fmt(gap, 2, signed=True)} vs WS</span>"
            cells.append(f"<td class='num'>{w.fmt(shown, digits)}{extra}</td>")
        body.append(f"<tr class='{emphasis}'><td>{html.escape(label)}</td>{''.join(cells)}</tr>")
    return f"<div class='tw'><table><tr><th>wariant</th>{head}</tr>{''.join(body)}</table></div>"


def view_strip(scene, view, index, runs, labels, ws_run):
    target = w.load_rgb(DATASETS / scene / "images" / f"{view}.png")
    ws = w.load_rgb(ws_run / "renders/rgb" / f"{view}.png")
    pictures = [("GT", target), ("WaterSplatting", ws)]
    for label, run in zip(labels, runs):
        folder = run / "joint/evaluation" / f"iteration_{w.best_or_last(run, 'last'):06d}"
        pictures.append((label, w.load_rgb(folder / f"{index:04d}_{view}_hybrid.png")))
    best = pictures[-1][1]
    figures = [(label, Image.fromarray((image * 255).round().astype(np.uint8))) for label, image in pictures]
    figures.append((f"{labels[-1]} vs WS: niebieski = my lepiej", w.difference_map(best, ws, target)))
    return (f"<h4>{view}</h4><div class='grid'>" + "".join(
        f"<figure><img loading='lazy' src='{thumbnail(image, 440)}' alt=''><figcaption>{html.escape(label)}"
        f"</figcaption></figure>" for label, image in figures) + "</div>")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True, help="sweep holding the control (no suffix)")
    parser.add_argument("--ws", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    masks = load_masks(ROOT / "configs/water_sam")
    seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    panels, gains, variants, summary = [], {}, [], {}
    for scene in PROMPTS:
        control_run = args.control / (scene + seed_suffix)
        ws_run = args.ws / scene
        if not control_run.is_dir() or not (ws_run / "renders/rgb").is_dir():
            continue
        ws_images = {view: w.load_rgb(ws_run / "renders/rgb" / f"{view}.png") for view in PROMPTS[scene]}
        ws_values = score_views(scene, ws_images, masks)
        ws_values.update(fps=json.loads((ws_run / "renders/render_eval.json").read_text()).get("render_fps",
                                                                                                 float("nan")),
                         gaussians=json.loads((ws_run / "renders/render_eval.json").read_text()).get(
                             "num_gaussians", float("nan")) / 1000)
        control = collect(control_run, masks)
        runs = sorted(p for p in args.sweep.iterdir() if p.name.startswith(scene)
                      and p.name.endswith(seed_suffix or "") and (p / "joint/metrics/eval.csv").is_file()
                      and ("_seed" not in p.name[len(scene):] or seed_suffix))
        results = [(p.name[len(scene):].removesuffix(seed_suffix) or "(brak)", p, collect(p, masks)) for p in runs]
        for name, _, values in results:
            if name not in variants:
                variants.append(name)
            gains[(scene, name)] = [values["all"] - control["all"]]
        best = max(results, key=lambda r: r[2]["all"])
        summary[scene] = {"WaterSplatting": ws_values, "control": control, **{n: v for n, _, v in results}}
        rows = ([("WaterSplatting", ws_values, "ws"), ("kontrola (default)", control, "base")]
                + [(name, values, "pick" if name == best[0] else "") for name, _, values in results])
        strips = "".join(view_strip(scene, view, i, [control_run, best[1]], ["kontrola", f"najlepszy: {best[0]}"],
                                    ws_run) for i, view in enumerate(PROMPTS[scene]))
        panels.append((SHORT[scene], scene_table(rows, control, ws_values) + "<h3>Widoki testowe</h3>" + strips))
    categories = [(scene, SHORT[scene]) for scene in PROMPTS if any(k[0] == scene for k in gains)]
    gain_chart = "".join(chart(f"Zysk PSNR względem kontroli [dB]: {variant}", categories, [("", "")],
                               {(scene, ""): gains[(scene, variant)] for scene, _ in categories
                                if (scene, variant) in gains}, 2, as_points=True, unit=" dB")
                         for variant in variants)
    tabs = "".join(f'<button class="tab" data-tab="{i}" aria-pressed="{str(i == 0).lower()}">{html.escape(l)}</button>'
                   for i, (l, _) in enumerate(panels))
    bodies = "".join(f'<div class="panel" data-tab="{i}"{"" if i == 0 else " hidden"}>{b}</div>'
                     for i, (_, b) in enumerate(panels))
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in args.note)
    body = f"""
<h1>Warianty względem kontroli i WaterSplatting</h1>
<p class="lead">Sweep {html.escape(str(args.sweep))}, kontrola {html.escape(str(args.control))}, seed {args.seed}.
Wszystko z ostatniej ewaluacji (jak WaterSplatting: końcowy checkpoint), z zapisanych renderów 8-bit; dno i woda
według masek SAM; „kolor/mgła” = MSE błędu rozmytego σ = {LOW_SIGMA} px, „detal” = reszta.</p>
<ul class="small">{notes}</ul>
<div class="charts">{gain_chart}</div>
<div class="tabs">{tabs}</div>{bodies}"""
    style = """<style>tr.ws td{font-weight:600} tr.base td{font-style:italic} td .up{color:var(--up)} td .down{color:var(--down)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:12px;margin:12px 0}
.viz{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px} .chart{width:100%}
.grid{stroke:var(--line)} .tick{fill:var(--muted);font-size:11px} .dot.s1{fill:var(--accent);stroke:var(--panel);stroke-width:2}
.range.s1{stroke:var(--accent);stroke-width:2} .hit{fill:transparent} figcaption{font-weight:400}
div.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:8px}</style>"""
    page = w.TEMPLATE.replace("{{BODY}}", body).replace("</head>", style + "</head>").replace(
        "<title>FoamSurf vs WaterSplatting</title>", "<title>Warianty vs WaterSplatting</title>")
    args.output.write_text(page)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(args.output)


if __name__ == "__main__":
    main()
