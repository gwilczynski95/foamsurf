#!/usr/bin/env python3
"""HTML reports for runs of scripts/train_seathru.sbatch.

  python scripts/seathru_report.py scenes --sweep outputs/seathru/<job> --output report.html
  python scripts/seathru_report.py cells  --sweep outputs/seathru/<job> --output report.html

scenes: the configuration on every SeaThru-NeRF scene (averaged over seeds), dataset
        means, FPS, water/seabed metrics on the SAM masks, images of every test view.
cells:  Voronoi layout x colour smoothing per scene; medium-only and hybrid renders of
        one test view per scene at the last iteration (where the cells show most).

Numbers: best test PSNR/SSIM at the best iteration; FPS = median hybrid FPS of the last
10 evaluations (full-resolution test views on an RTX 4090, the start-cell lookup included);
water/seabed coverage from the Gaussian-only alpha maps at the best iteration on
configs/water_sam (scripts/medium_fix_report.py).
"""
import argparse
import csv
import html
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medium_fix_report as base  # noqa: E402
from seabed_focus_report import COLORS, coverage_map, fmt, thumbnail  # noqa: E402

SCENES = ("IUI3-RedSea-Undistorted", "Curasao-Undistorted", "JapaneseGradens-RedSea-Undistorted",
          "Panama-Undisorted")
SHORT = {"IUI3-RedSea-Undistorted": "IUI3 Red Sea", "Curasao-Undistorted": "Curaçao",
         "JapaneseGradens-RedSea-Undistorted": "Japanese Gardens", "Panama-Undisorted": "Panama"}
LAYOUTS = (("", "dziś: 5k szeroko"), ("_view75", "5k, 75% w polu widzenia"),
           ("_view75_c20k", "20k, 75% w polu widzenia"))
SMOOTHING = (("", "1"), ("_cs5", "5"), ("_cs10", "10"))
CELL_VIEWS = {"IUI3-RedSea-Undistorted": "MTN_5894", "Panama-Undisorted": "MTN_1529"}
METRICS = (  # key, label, digits, percent
    ("best_psnr", "best PSNR [dB]", 2, False),
    ("final_psnr", "PSNR na końcu [dB]", 2, False),
    ("ssim", "SSIM", 3, False),
    ("fps", "FPS (4090)", 0, False),
    ("gaussians", "Gaussy [tys.]", 0, False),
    ("cells", "komórki", 0, False),
    ("minutes", "trening [min]", 0, False),
    ("psnr_water", "PSNR wody [dB]", 2, False),
    ("sam_water_gt_0.5", "woda zajęta przez Gaussy", 2, True),
    ("sam_seabed_gt_0.99", "całe dno α>0,99", 1, True),
    ("sam_seabed_0_30_gt_0.5", "dno 0–30 px od wody", 1, True),
    ("sam_seabed_30_60_gt_0.5", "dno 30–60 px od wody", 1, True),
    ("delta_near", "ΔPSNR dna po czyszczeniu komórek [dB]", 2, False),
)


def split_name(name):
    """(scene, variant suffix, seed) of a run directory name."""
    scene = next(s for s in SCENES if name.startswith(s))
    rest = name[len(scene):]
    seed = 42
    match = re.search(r"_seed(\d+)$", rest)
    if match:
        seed, rest = int(match.group(1)), rest[:match.start()]
    return scene, rest, seed


def fps_median(path, last=10):
    rows = [r for r in csv.DictReader((path / "joint/metrics/eval.csv").open()) if r.get("eval/hybrid_fps")]
    values = sorted(float(r["eval/hybrid_fps"]) for r in rows[-last:])
    return values[len(values) // 2] if values else float("nan")


def load(path):
    run = base.load_run(path)
    scene, variant, seed = split_name(path.name)
    rows = {int(float(r["iteration"])): r for r in csv.DictReader((path / "joint/metrics/eval.csv").open())
            if r.get("eval/hybrid_psnr")}
    summary = json.loads((path / "joint/summary.json").read_text())
    values = dict(best_psnr=run.get("best_psnr", float("nan")), final_psnr=run.get("final_psnr", float("nan")),
                  ssim=float(rows[run["best_iteration"]]["eval/hybrid_ssim"]), fps=fps_median(path),
                  gaussians=summary["gaussian_count"] / 1000, cells=summary["voronoi_count"],
                  minutes=summary["training_seconds"] / 60,
                  psnr_water=base.audit_value(run, "original/psnr_water"),
                  delta_near=base.audit_value(run, "clean_near/delta_psnr_bottom"))
    for key, *_ in METRICS:
        values.setdefault(key, base.audit_value(run, key))
    return dict(run=run, scene=scene, variant=variant, seed=seed, values=values,
                last_iteration=max(rows))


def mean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")


def group(runs):
    """(scene, variant) -> list of runs (seeds)."""
    groups = {}
    for run in runs:
        groups.setdefault((run["scene"], run["variant"]), []).append(run)
    return groups


def show(value, digits, percent):
    return fmt(100 * value if percent else value, digits) + ("%" if percent and not math.isnan(value) else "")


def chart(title, categories, series, values, digits, as_points=False, unit=""):
    """categories: [(key, label)]; series: [(key, label)] (<= 3, fixed colour slots);
    values[(category, series)] = [per-seed values]. Bars from zero, or mean markers with
    the seed range for interval scales such as PSNR."""
    flat = [v for seeds in values.values() for v in seeds if not math.isnan(v)]
    if not flat:
        return ""
    low, high = (min(flat), max(flat)) if as_points else (0.0, max(flat))
    step = base.nice_step(max(high - low, 1e-6))
    low = math.floor(low / step) * step - (step if as_points else 0)
    high = math.ceil(high / step) * step
    width, height, left, bottom, top = 560, 240, 46, 40, 12
    plot_w, plot_h = width - left - 8, height - top - bottom
    sy = lambda v: top + plot_h - (v - low) / max(high - low, 1e-9) * plot_h  # noqa: E731
    group_w = plot_w / len(categories)
    bar_w = min(26, group_w / (len(series) + 1))
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(title)}">']
    tick = low
    while tick <= high + 1e-9:
        parts.append(f'<line class="grid" x1="{left}" x2="{width - 8}" y1="{sy(tick):.1f}" y2="{sy(tick):.1f}"/>'
                     f'<text class="tick" x="{left - 5}" y="{sy(tick) + 4:.1f}" text-anchor="end">'
                     f'{fmt(tick, 1 if step < 1 else 0)}</text>')
        tick += step
    for index, (category, category_label) in enumerate(categories):
        center = left + group_w * (index + 0.5)
        start = center - bar_w * len(series) / 2
        for slot, (key, label) in enumerate(series, 1):
            seeds = values.get((category, key))
            if not seeds:
                continue
            x, value = start + (slot - 1) * bar_w, mean(seeds)
            middle = x + bar_w / 2
            tip = (f"{category_label}, {label}: {fmt(value, digits)}{unit}"
                   + (f" (seedy: {', '.join(fmt(v, digits) for v in seeds)})" if len(seeds) > 1 else ""))
            if as_points:
                spread = [sy(v) for v in seeds if not math.isnan(v)]
                mark = (f'<line class="range s{slot}" x1="{middle:.1f}" x2="{middle:.1f}" y1="{min(spread):.1f}" '
                        f'y2="{max(spread):.1f}"/><circle class="dot s{slot}" cx="{middle:.1f}" cy="{sy(value):.1f}" r="6"/>')
            else:
                y0, y1 = sy(low), sy(value)
                r = min(4.0, max(y0 - y1, 0) / 2)
                w = bar_w - 2
                mark = (f'<path class="bar s{slot}" d="M{x:.1f},{y0:.1f} V{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} '
                        f'H{x + w - r:.1f} Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 + r:.1f} V{y0:.1f} Z"/>')
            parts.append(f'<g class="mark">{mark}<rect class="hit" x="{x:.1f}" y="{top}" width="{bar_w:.1f}" '
                         f'height="{plot_h}"><title>{html.escape(tip)}</title></rect></g>')
            if len(seeds) > 1:
                parts.extend(f'<circle class="seed" cx="{middle:.1f}" cy="{sy(v):.1f}" r="2.6"/>' for v in seeds)
        parts.append(f'<text class="tick" x="{center:.1f}" y="{height - bottom + 16}" text-anchor="middle">'
                     f'{html.escape(category_label)}</text>')
    parts.append("</svg>")
    legend = "" if len(series) < 2 else '<div class="legend">' + "".join(
        f'<span class="key"><i class="sw s{slot}"></i>{html.escape(label)}</span>'
        for slot, (_, label) in enumerate(series, 1)) + "</div>"
    return f'<figure class="viz"><figcaption>{html.escape(title)}</figcaption>{legend}{"".join(parts)}</figure>'


def table(rows, header_label, highlight=None):
    """rows: [(label, [runs])] -> metric table of seed means (seeds on hover)."""
    head = "".join(f"<th class='num'>{html.escape(label)}</th>" for _, label, *_ in METRICS)
    body = []
    for label, members in rows:
        cells = []
        for key, _, digits, percent in METRICS:
            seeds = [m["values"][key] for m in members]
            tip = " / ".join(show(v, digits, percent) for v in seeds)
            cells.append(f"<td class='num' title='seedy: {tip}'>{show(mean(seeds), digits, percent)}</td>")
        cls = " class='pick'" if highlight and label in highlight else ""
        body.append(f"<tr{cls}><td>{label}</td>{''.join(cells)}</tr>")
    return (f"<div class='tw'><table><tr><th>{html.escape(header_label)}</th>{head}</tr>"
            f"{''.join(body)}</table></div>")


def image_row(label, run, view, index, columns, iteration=None):
    iteration = iteration or run["run"]["best_iteration"]
    folder = run["run"]["path"] / "joint/evaluation" / f"iteration_{iteration:06d}"
    dataset = json.loads((run["run"]["path"] / "pipeline_config.json").read_text())["joint"]["dataset"]["source_path"]
    target = Image.open(Path(dataset) / "images" / f"{view}.png")
    cells = []
    for kind in columns:
        if kind == "gt":
            source = thumbnail(target)
        elif kind == "coverage":
            path = folder / f"{index:04d}_{view}_gaussian_alpha.npz"
            water = base.water_regions().get(view, {}).get("water")
            source = (thumbnail(coverage_map(target, np.load(path)["alpha"][..., 0], water))
                      if path.is_file() and water is not None else None)
        else:
            path = folder / f"{index:04d}_{view}_{kind}.png"
            source = thumbnail(Image.open(path)) if path.is_file() else None
        cells.append(f'<td><img loading="lazy" src="{source}" alt=""></td>' if source else "<td>–</td>")
    return f"<tr><th>{label}</th>{''.join(cells)}</tr>"


COLUMN_LABELS = dict(gt="GT", hybrid="hybryda", gaussian_only_black="Gaussy na czarnym",
                     coverage="pokrycie (granica SAM)", voronoi_only="samo medium", backscatter="backscatter")


def test_views(run):
    dataset = json.loads((run["run"]["path"] / "pipeline_config.json").read_text())["joint"]["dataset"]["source_path"]
    from sam_water_mask import PROMPTS
    return list(PROMPTS.get(Path(dataset).name, {}))


def tabs(panels):
    """panels: [(label, html)] -> tab bar + panels (first visible)."""
    buttons = "".join(f'<button class="tab" data-tab="{i}" aria-pressed="{str(i == 0).lower()}">{html.escape(l)}</button>'
                      for i, (l, _) in enumerate(panels))
    bodies = "".join(f'<div class="panel" data-tab="{i}"{"" if i == 0 else " hidden"}>{b}</div>'
                     for i, (_, b) in enumerate(panels))
    return f'<div class="tabs">{buttons}</div>{bodies}'


def coverage_legend():
    return '<div class="legend">' + "".join(
        f'<span class="key"><i style="background:rgb{c}"></i>{t}</span>' for c, t in (
            (COLORS["covered"], "dno z Gaussami (α>0,5)"), (COLORS["empty"], "dno bez Gaussów (medium)"),
            (COLORS["water"], "woda zajęta przez Gaussy"), ((255, 255, 255), "granica woda/dno wg SAM"))) + "</div>"


def scenes_report(runs, sources, notes):
    groups = group(runs)
    scenes = [s for s in SCENES if (s, "") in groups]
    columns = ("gt", "hybrid", "gaussian_only_black", "coverage", "voronoi_only")
    per_scene = {s: groups[(s, "")] for s in scenes}
    dataset_means = {key: mean([mean([m["values"][key] for m in per_scene[s]]) for s in scenes]) for key, *_ in METRICS}
    tiles = "".join(
        f"<div class='tile'><div class='small'>{html.escape(label)}</div><div class='hero'>"
        f"{show(dataset_means[key], digits, percent)}</div></div>"
        for key, label, digits, percent in METRICS if key in ("best_psnr", "ssim", "fps", "gaussians", "minutes",
                                                             "sam_water_gt_0.5", "sam_seabed_0_30_gt_0.5"))
    size = {}
    for s in scenes:
        image = Image.open(Path(json.loads((per_scene[s][0]["run"]["path"] / "pipeline_config.json").read_text())
                                ["joint"]["dataset"]["source_path"]) / "images" / f"{test_views(per_scene[s][0])[0]}.png")
        size[s] = f"{image.width}×{image.height}"
    rows = [(f"{SHORT[s]}<br><span class='small'>{size[s]}, {len(per_scene[s])} seedy</span>", per_scene[s])
            for s in scenes]
    categories = [(s, SHORT[s]) for s in scenes]
    one = [("", "")]
    charts = (chart("best PSNR [dB] (punkt: średnia, kreska: seedy)", categories, one,
                    {(s, ""): [m["values"]["best_psnr"] for m in per_scene[s]] for s in scenes}, 2, as_points=True,
                    unit=" dB")
              + chart("FPS na RTX 4090 (pełna rozdzielczość)", categories, one,
                      {(s, ""): [m["values"]["fps"] for m in per_scene[s]] for s in scenes}, 0)
              + chart("Dno przy wodzie pokryte Gaussami (α>0,5) [%]", categories,
                      [("0_30", "0–30 px od wody"), ("30_60", "30–60 px od wody")],
                      {(s, ring): [100 * m["values"][f"sam_seabed_{ring}_gt_0.5"] for m in per_scene[s]]
                       for s in scenes for ring in ("0_30", "30_60")}, 1)
              + chart("Woda zajęta przez Gaussy [%]", categories, one,
                      {(s, ""): [100 * m["values"]["sam_water_gt_0.5"] for m in per_scene[s]] for s in scenes}, 2))
    head = "".join(f"<th>{COLUMN_LABELS[c]}</th>" for c in columns)
    panels = []
    for s in scenes:
        run = next(m for m in per_scene[s] if m["seed"] == 42)
        body = "".join(image_row(view, run, view, i, columns) for i, view in enumerate(test_views(run)))
        panels.append((SHORT[s], f"<p class='small'>seed 42, najlepsza iteracja {run['run']['best_iteration']}</p>"
                                 f"<div class='tw'><table class='images'><tr><th></th>{head}</tr>{body}</table></div>"))
    content = f"""
<h1>FoamSurf na SeaThru-NeRF: domyślna konfiguracja na wszystkich scenach</h1>
<p class="lead">Konfiguracja <code>configs/iui3_default.yaml</code> (chroma na wszystkich promieniach, densyfikacja
kompensowana tłumieniem, pokrycie dna α&gt;0,9, strata wodna 1e-2, do 700 tys. Gaussów) na czterech scenach
Undistorted, seedy 42 i 43. Wartości: średnie z seedów (najedź na komórkę tabeli, żeby zobaczyć seedy).</p>
<h2>1. Cały dataset (średnia ze scen)</h2>
<div class="tiles">{tiles}</div>
<ul class="small">{notes}</ul>
<h2>2. Sceny</h2>
{table(rows, "scena")}
<div class="charts">{charts}</div>
<h2>3. Obrazy widoków testowych</h2>
{coverage_legend()}
{tabs(panels)}
<p class="small">FPS: mediana z 10 ostatnich ewaluacji (render hybrydy pełnej rozdzielczości, RTX 4090, z wyszukaniem
komórki startowej). Woda i dno: maski SAM (configs/water_sam); Panama MTN_1539 i MTN_1547 nie mają otwartej wody, więc
liczą się tylko w metrykach dna. Źródła: {html.escape(sources)}</p>"""
    return page("SeaThru: wszystkie sceny", content)


def cells_report(runs, sources, notes):
    groups = group(runs)
    scenes = [s for s in SCENES if any(key[0] == s for key in groups)]
    panels = []
    for s in scenes:
        def members(layout, smoothing):
            return groups.get((s, layout + smoothing), [])
        rows = [(f"{layout_label}, wygładzanie {smoothing_label}", members(layout, smoothing))
                for layout, layout_label in LAYOUTS for smoothing, smoothing_label in SMOOTHING
                if members(layout, smoothing)]
        categories = [(smoothing, f"wygładzanie {label}") for smoothing, label in SMOOTHING]
        def values(key, scale=1.0):
            return {(smoothing, layout): [scale * m["values"][key] for m in members(layout, smoothing)]
                    for layout, _ in LAYOUTS for smoothing, _ in SMOOTHING if members(layout, smoothing)}
        charts = (chart("best PSNR [dB]", categories, LAYOUTS, values("best_psnr"), 2, as_points=True, unit=" dB")
                  + chart("PSNR wody [dB]", categories, LAYOUTS, values("psnr_water"), 2, as_points=True, unit=" dB")
                  + chart("FPS na RTX 4090", categories, LAYOUTS, values("fps"), 0)
                  + chart("Dno 0–30 px od wody pokryte Gaussami (α>0,5) [%]", categories, LAYOUTS,
                          values("sam_seabed_0_30_gt_0.5", 100), 1)
                  + chart("Woda zajęta przez Gaussy [%]", categories, LAYOUTS, values("sam_water_gt_0.5", 100), 2)
                  + chart("ΔPSNR dna po wyczyszczeniu komórek [dB] (bliżej 0 = medium mniej podpiera dno)",
                          categories, LAYOUTS, values("delta_near"), 2, as_points=True, unit=" dB"))
        view = CELL_VIEWS.get(s)
        grids = ""
        if view:
            for kind, title in (("voronoi_only", "Samo medium (Voronoi-only), ostatnia iteracja"),
                                ("hybrid", "Hybryda, ostatnia iteracja")):
                header = "".join(f"<th>wygładzanie {label}</th>" for _, label in SMOOTHING)
                lines = []
                for layout, layout_label in LAYOUTS:
                    cells = []
                    for smoothing, _ in SMOOTHING:
                        found = members(layout, smoothing)
                        if not found:
                            cells.append("<td>–</td>")
                            continue
                        run = found[0]
                        index = test_views(run).index(view)
                        path = (run["run"]["path"] / "joint/evaluation" / f"iteration_{run['last_iteration']:06d}"
                                / f"{index:04d}_{view}_{kind}.png")
                        cells.append(f'<td><img loading="lazy" src="{thumbnail(Image.open(path), 520)}" alt=""></td>'
                                     if path.is_file() else "<td>–</td>")
                    lines.append(f"<tr><th>{html.escape(layout_label)}</th>{''.join(cells)}</tr>")
                grids += (f"<h3>{title}: {view}</h3><div class='tw'><table class='images'><tr><th></th>{header}</tr>"
                          f"{''.join(lines)}</table></div>")
        body = (f"<div class='charts'>{charts}</div>{grids}<h3>Wszystkie warianty</h3>"
                f"{table(rows, 'wariant', highlight={rows[0][0]})}")
        panels.append((SHORT[s], body))
    content = f"""
<h1>Komórki Voronoi: rozkład × wygładzanie koloru</h1>
<p class="lead">Na domyślnej konfiguracji: trzy rozkłady komórek (dziś 5k szeroko wokół sceny; 5k z 75% w objętości
widzianej przez kamery; 20k z 75% w tej objętości) × wygładzanie koloru komórek 1 / 5 / 10, seed 42. Pytanie: czy
gęstsze komórki w polu widzenia usuwają widoczne komórki bez oddawania dna medium i bez spadku PSNR.</p>
<ul class="small">{notes}</ul>
<div class="legend"><span class="key"><i class="sw s1"></i>{LAYOUTS[0][1]}</span>
<span class="key"><i class="sw s2"></i>{LAYOUTS[1][1]}</span><span class="key"><i class="sw s3"></i>{LAYOUTS[2][1]}</span></div>
{tabs(panels)}
<p class="small">Jeden seed na wariant: różnice PSNR poniżej ok. 0,3 dB są w granicach szumu seedów. FPS: mediana z 10
ostatnich ewaluacji (RTX 4090, pełna rozdzielczość). Źródła: {html.escape(sources)}</p>"""
    return page("Komórki Voronoi", content)


def page(title, content):
    return TEMPLATE.replace("{{TITLE}}", html.escape(title)).replace("{{CONTENT}}", content)


TEMPLATE = """<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{{TITLE}}</title>
<style>
:root{--bg:#fcfcfb;--panel:#fff;--text:#0b0b0b;--muted:#52514e;--line:#e4e3df;--grid:#ecebe7;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--pick:#eef4fc}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--panel:#222220;--text:#f0efec;
--muted:#c3c2b7;--line:#3a3936;--grid:#2e2d2b;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--pick:#1f2a38}}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#222220;--text:#f0efec;--muted:#c3c2b7;--line:#3a3936;--grid:#2e2d2b;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--pick:#1f2a38}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}
main{max-width:1280px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.55rem;margin:0 0 6px} h2{font-size:1.2rem;margin:34px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:1rem;margin:22px 0 6px} .small{font-size:.8rem;color:var(--muted)} .lead{color:var(--muted);max-width:920px}
code{font-size:.85em}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.hero{font-size:1.6rem;font-weight:600;font-variant-numeric:tabular-nums}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:14px;margin:14px 0}
.viz{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}
figcaption{font-weight:600;margin-bottom:4px} .chart{width:100%;height:auto}
.grid{stroke:var(--grid);stroke-width:1} .tick{fill:var(--muted);font-size:11px}
.bar.s1,.dot.s1,.sw.s1{fill:var(--s1);background:var(--s1)} .bar.s2,.dot.s2,.sw.s2{fill:var(--s2);background:var(--s2)}
.bar.s3,.dot.s3,.sw.s3{fill:var(--s3);background:var(--s3)}
.dot{stroke:var(--panel);stroke-width:2} .range{stroke-width:2} .range.s1{stroke:var(--s1)} .range.s2{stroke:var(--s2)}
.range.s3{stroke:var(--s3)} .seed{fill:var(--text);stroke:var(--panel);stroke-width:1.5} .hit{fill:transparent}
.mark:hover .bar,.mark:hover .dot{opacity:.8}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0;font-size:.85rem}
.key i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px;border:1px solid var(--line)}
.tw{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:.84rem}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums} tr.pick td{background:var(--pick)}
.images img{width:100%;min-width:200px;border-radius:4px;display:block} .images th{min-width:90px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}
.tab{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:6px;padding:4px 12px;cursor:pointer}
.tab[aria-pressed="true"]{border-color:var(--s1);color:var(--s1);font-weight:600}
</style></head><body><main>{{CONTENT}}</main>
<script>
document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(t => t.setAttribute('aria-pressed', String(t === tab)));
  document.querySelectorAll('.panel').forEach(p => p.hidden = p.dataset.tab !== tab.dataset.tab);
}));
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("scenes", "cells"))
    parser.add_argument("--sweep", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    runs = [load(path) for sweep in args.sweep for path in sorted(sweep.iterdir())
            if (path / "joint/summary.json").is_file()]
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in args.note)
    sources = ", ".join(str(s) for s in args.sweep)
    text = scenes_report(runs, sources, notes) if args.mode == "scenes" else cells_report(runs, sources, notes)
    args.output.write_text(text)
    print(args.output)


if __name__ == "__main__":
    main()
