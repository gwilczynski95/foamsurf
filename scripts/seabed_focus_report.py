#!/usr/bin/env python3
"""Focused HTML report: what improves seabed coverage and the water/seabed boundary.

Usage: python scripts/seabed_focus_report.py --sweep outputs/iui3_medium_fix/<job> --baseline <variant>
       [--output report.html] [--top 3]

Every water/seabed number uses the SAM masks of the test views (configs/water_sam/<scene>.npz)
and the Gaussian-only alpha maps saved at each run's best iteration (see
scripts/medium_fix_report.py, whose loaders and caches this script reuses). Variants are
averaged over seeds. Factor charts expect the chroma / compensation naming of the
'chroma' sweep set (B0_chroma<W>[_chthr02]..._g700k[_tden]_water1e2_bk); other names
appear in the tables only.

Most promising runs (stated in the report): water taken by Gaussians <= MAX_WATER and best
PSNR >= baseline - PSNR_SLACK, ranked by the mean seabed coverage 0-30 and 30-60 px
from the water.
"""
import argparse
import base64
import html
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medium_fix_report as base  # noqa: E402

MAX_WATER = 0.005
PSNR_SLACK = 0.1
VIEW_FOR_GRID = None  # all four test views get a tab
METRICS = (  # key, label, scale, digits, higher is better
    ("best_psnr", "best PSNR [dB]", 1, 2, True),
    ("psnr_water", "PSNR wody [dB]", 1, 2, True),
    ("sam_water_gt_0.5", "woda zajęta przez Gaussy [%]", 100, 2, False),
    ("sam_seabed_0_30_gt_0.5", "dno 0–30 px od wody, α>0,5 [%]", 100, 1, True),
    ("sam_seabed_30_60_gt_0.5", "dno 30–60 px od wody, α>0,5 [%]", 100, 1, True),
    ("sam_seabed_60_120_gt_0.5", "dno 60–120 px od wody, α>0,5 [%]", 100, 1, True),
    ("sam_seabed_gt_0.99", "całe dno α>0,99 [%]", 100, 1, True),
    ("delta_near", "ΔPSNR dna po czyszczeniu komórek [dB]", 1, 2, True),
    ("gaussians", "Gaussy [tys.]", 0.001, 0, None),
)
COLORS = dict(covered=(27, 175, 122), empty=(235, 104, 52), water=(98, 80, 214))


def run_value(run, key):
    if key == "best_psnr":
        return run.get("best_psnr", float("nan"))
    if key == "psnr_water":
        return base.audit_value(run, "original/psnr_water")
    if key == "delta_near":
        return base.audit_value(run, "clean_near/delta_psnr_bottom")
    if key == "gaussians":
        return base.gaussian_count(run)
    return base.audit_value(run, key)


def summarize(runs):
    """Variant -> {key: [value per seed]} plus the runs, in first-seen order."""
    variants = {}
    for run in runs:
        entry = variants.setdefault(run["variant"], dict(runs=[], values={key: [] for key, *_ in METRICS}))
        entry["runs"].append(run)
        for key, *_ in METRICS:
            entry["values"][key].append(run_value(run, key))
    return variants


def mean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")


def boundary_score(entry):
    return (mean(entry["values"]["sam_seabed_0_30_gt_0.5"]) + mean(entry["values"]["sam_seabed_30_60_gt_0.5"])) / 2


def promising(variants, baseline, top):
    """Variants that keep water clean and PSNR, ranked by boundary coverage."""
    reference = mean(variants[baseline]["values"]["best_psnr"])
    eligible = [name for name, entry in variants.items() if name != baseline
                and mean(entry["values"]["sam_water_gt_0.5"]) <= MAX_WATER
                and mean(entry["values"]["best_psnr"]) >= reference - PSNR_SLACK]
    return sorted(eligible, key=lambda name: -boundary_score(variants[name]))[:top]


def factors(variant):
    """(chroma setting label, compensated) for the chroma sweep naming, or None."""
    match = re.fullmatch(r"B0_chroma(\d{3,4})(_chthr02)?_.*?(_tden)?_water1e2_bk", variant)
    if not match:
        return None
    weight = int(match.group(1)) / 100
    label = f"{weight:g}".replace(".", ",") + (" (wszystkie promienie)" if match.group(2) else " (α>0,5)")
    return label, bool(match.group(3))


def short_label(variant, baseline):
    """What differs from the baseline, in a few words."""
    if variant == baseline:
        return "baza"
    parsed = factors(variant)
    if parsed is None:
        return base.describe(variant)
    chroma, compensated = parsed
    weight, rays = chroma.split(" ", 1)
    text = f"chroma {weight} " + ("na wszystkich promieniach" if "wszystkie" in rays else "tylko dla α>0,5")
    return text + (" + kompensacja densyfikacji" if compensated else "")


def fmt(value, digits):
    return "–" if math.isnan(value) else f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def factor_chart(variants, key, label, scale, digits):
    """x = chroma setting, two series = without / with compensation, small dots = seeds.

    Percentages are bars from zero; PSNR (an interval scale) is a mean marker with the
    seed range, so the zoomed axis does not exaggerate differences.
    """
    groups = {}
    for name, entry in variants.items():
        parsed = factors(name)
        if parsed:
            groups.setdefault(parsed[0], {})[parsed[1]] = [v * scale for v in entry["values"][key]]
    if not groups:
        return ""
    order = sorted(groups, key=lambda g: (" (α>0,5)" not in g, float(g.split(" ")[0].replace(",", "."))))
    values = [v for g in groups.values() for seeds in g.values() for v in seeds if not math.isnan(v)]
    low, high = min(0.0, min(values)), max(values)
    as_points = key in ("best_psnr", "psnr_water")
    if as_points:
        step = base.nice_step(high - min(values))
        low, high = math.floor(min(values) / step) * step - step, math.ceil(high / step) * step
    else:
        step = base.nice_step(high - low)
        high = math.ceil(high / step) * step
    width, height, left, bottom, top = 520, 250, 46, 64, 12
    plot_w, plot_h = width - left - 8, height - top - bottom
    sy = lambda v: top + plot_h - (v - low) / max(high - low, 1e-9) * plot_h  # noqa: E731
    group_w = plot_w / len(order)
    bar_w = min(26, group_w / 3)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(label)}">']
    tick = low
    while tick <= high + 1e-9:
        parts.append(f'<line class="grid" x1="{left}" x2="{width - 8}" y1="{sy(tick):.1f}" y2="{sy(tick):.1f}"/>'
                     f'<text class="tick" x="{left - 5}" y="{sy(tick) + 4:.1f}" text-anchor="end">{fmt(tick, 1 if step < 1 else 0)}</text>')
        tick += step
    for index, group in enumerate(order):
        center = left + group_w * (index + 0.5)
        for slot, compensated in ((1, False), (2, True)):
            seeds = groups[group].get(compensated)
            if not seeds:
                continue
            x = center + (-bar_w - 1 if not compensated else 1)
            value = mean(seeds)
            tip = (f"{group}, {'z kompensacją' if compensated else 'bez kompensacji'}: {fmt(value, digits)} "
                   f"(seedy: {', '.join(fmt(v, digits) for v in seeds)})")
            middle = x + bar_w / 2
            if as_points:
                spread = [sy(v) for v in seeds if not math.isnan(v)]
                mark = (f'<line class="range s{slot}" x1="{middle:.1f}" x2="{middle:.1f}" y1="{min(spread):.1f}" '
                        f'y2="{max(spread):.1f}"/><circle class="dot s{slot}" cx="{middle:.1f}" cy="{sy(value):.1f}" r="6"/>')
            else:
                y0, y1 = sy(low), sy(value)
                r = min(4.0, max(y0 - y1, 0) / 2)
                mark = (f'<path class="bar s{slot}" d="M{x:.1f},{y0:.1f} V{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} '
                        f'H{x + bar_w - r:.1f} Q{x + bar_w:.1f},{y1:.1f} {x + bar_w:.1f},{y1 + r:.1f} V{y0:.1f} Z"/>')
            parts.append(f'<g class="mark">{mark}<rect class="hit" x="{x - 2:.1f}" y="{top}" width="{bar_w + 4:.1f}" '
                         f'height="{plot_h}"><title>{html.escape(tip)}</title></rect></g>')
            for seed in seeds:
                parts.append(f'<circle class="seed" cx="{middle:.1f}" cy="{sy(seed):.1f}" r="2.6"/>')
        weight, *rest = group.split(" ", 1)
        baseline_group = "α>0,5" in group
        parts.append(f'<text class="tick" x="{center:.1f}" y="{height - bottom + 16}" text-anchor="middle">{weight}</text>'
                     + (f'<text class="tick small" x="{center:.1f}" y="{height - bottom + 30}" text-anchor="middle">'
                        f'tylko α&gt;0,5</text>' if baseline_group else ""))
    parts.append(f'<text class="tick" x="{left + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle">'
                 f'waga chromy (poza pierwszą grupą: na wszystkich promieniach)</text></svg>')
    return f'<figure class="viz"><figcaption>{html.escape(label)}</figcaption>{"".join(parts)}</figure>'


def thumbnail(image, width=440):
    image = image.convert("RGB")
    image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def boundary(mask):
    from sam_water_mask import boundary as edge
    return edge(mask)


def coverage_map(target, alpha, water):
    """Grayed ground truth with seabed covered / seabed left to the medium / water taken
    by Gaussians, and the SAM boundary in white."""
    gray = np.asarray(target.convert("L"), dtype=np.float32)[..., None].repeat(3, -1) * 0.55
    picture = gray.copy()
    seabed, opaque = ~water, alpha > 0.5
    for region, color in ((seabed & opaque, COLORS["covered"]), (seabed & ~opaque, COLORS["empty"]),
                          (water & opaque, COLORS["water"])):
        picture[region] = 0.35 * gray[region] + 0.65 * np.array(color)
    picture[boundary(water)] = 255
    return Image.fromarray(picture.clip(0, 255).astype(np.uint8))


def visuals(entries, dataset):
    """Tabs per test view; rows = runs, columns = GT, hybrid, Gaussians on black, coverage map."""
    regions = base.water_regions()
    tabs, panels = [], []
    for index, view in enumerate(base.VIEW_NAMES):
        target = Image.open(Path(dataset) / "images" / f"{view}.png")
        tabs.append(f'<button class="tab" data-view="{index}" aria-pressed="{str(index == 0).lower()}">{view}</button>')
        rows = []
        for label, run in entries:
            folder = run["path"] / "joint/evaluation" / f"iteration_{run['best_iteration']:06d}"
            cells = [thumbnail(target)]
            for kind in ("hybrid", "gaussian_only_black"):
                path = folder / f"{index:04d}_{view}_{kind}.png"
                cells.append(thumbnail(Image.open(path)) if path.is_file() else None)
            alpha_path = folder / f"{index:04d}_{view}_gaussian_alpha.npz"
            cells.append(thumbnail(coverage_map(target, np.load(alpha_path)["alpha"][..., 0], regions[view]["water"]))
                         if alpha_path.is_file() else None)
            images = "".join(f'<td><img loading="lazy" src="{src}" alt=""></td>' if src else "<td>–</td>" for src in cells)
            rows.append(f'<tr><th>{label}<br><span class="small">{html.escape(run["name"])}, it. '
                        f'{run["best_iteration"]}</span></th>{images}</tr>')
        panels.append(f'<div class="panel" data-view="{index}"{"" if index == 0 else " hidden"}><div class="tw">'
                      f'<table class="images"><tr><th></th><th>GT</th><th>hybryda</th><th>Gaussy na czarnym</th>'
                      f'<th>pokrycie (granica SAM)</th></tr>{"".join(rows)}</table></div></div>')
    legend = "".join(f'<span class="key"><i style="background:rgb{color}"></i>{text}</span>' for color, text in (
        (COLORS["covered"], "dno pokryte Gaussami (α>0,5)"), (COLORS["empty"], "dno bez Gaussów (robi je medium)"),
        (COLORS["water"], "woda zajęta przez Gaussy"), ((255, 255, 255), "granica woda/dno wg SAM")))
    return f'<div class="legend">{legend}</div><div class="tabs">{"".join(tabs)}</div>{"".join(panels)}'


def sam_masks(dataset):
    from sam_water_mask import PROMPTS, overlay
    cells = []
    prompts = PROMPTS.get(Path(dataset).name, {})  # the masks of this sweep's scene
    for view in prompts:
        region = base.water_regions()[view]
        image = np.asarray(Image.open(Path(dataset) / "images" / f"{view}.png").convert("RGB"))
        cells.append(f'<figure class="mask"><img loading="lazy" src="{thumbnail(overlay(image, region["water"], prompts[view]), 520)}" '
                     f'alt="maska {view}"><figcaption>{view}: woda {100 * region["water"].mean():.1f}% kadru</figcaption></figure>')
    return "".join(cells)


def metric_table(variants, names, baseline, highlight=()):
    head = "".join(f"<th class='num'>{html.escape(label)}</th>" for _, label, *_ in METRICS)
    reference = {key: mean(variants[baseline]["values"][key]) * scale for key, _, scale, _, _ in METRICS}
    body = []
    for name in names:
        entry = variants[name]
        cells = []
        for key, _, scale, digits, higher in METRICS:
            values = [v * scale for v in entry["values"][key]]
            value = mean(values)
            delta = value - reference[key]
            good = None if higher is None or name == baseline or abs(delta) < 10 ** -digits else (delta > 0) == higher
            mark = "" if good is None else (" up" if good else " down")
            seeds = " / ".join(fmt(v, digits) for v in values)
            cells.append(f"<td class='num{mark}' title='seedy: {seeds}'>{fmt(value, digits)}</td>")
        cls = " class='pick'" if name in highlight else (" class='base'" if name == baseline else "")
        body.append(f"<tr{cls}><td>{html.escape(short_label(name, baseline))}<br><span class='small'>{html.escape(name)}</span></td>"
                    f"{''.join(cells)}</tr>")
    return f"<div class='tw'><table><tr><th>wariant</th>{head}</tr>{''.join(body)}</table></div>"


def tiles(variants, baseline, winner):
    out = []
    for key, label, scale, digits, _ in METRICS[:7]:
        before = mean(variants[baseline]["values"][key]) * scale
        after = mean(variants[winner]["values"][key]) * scale
        out.append(f"<div class='tile'><div class='small'>{html.escape(label)}</div>"
                   f"<div class='hero'>{fmt(after, digits)}</div><div class='small'>baza: {fmt(before, digits)}</div></div>")
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", type=Path, action="append", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    runs = [base.load_run(path) for sweep in args.sweep for path in sorted(sweep.iterdir())
            if (path / "joint/metrics/eval.csv").is_file()]
    variants = summarize(runs)
    if args.baseline not in variants:
        sys.exit(f"baseline {args.baseline} not in the sweep")
    picks = promising(variants, args.baseline, args.top)
    dataset = json.loads((runs[0]["path"] / "pipeline_config.json").read_text())["joint"]["dataset"]["source_path"]
    first = lambda name: next(r for r in variants[name]["runs"] if r["seed"] is None)  # noqa: E731
    shown = [("baza", first(args.baseline))] + [(f"#{i + 1} {html.escape(short_label(name, args.baseline))}", first(name))
                                                for i, name in enumerate(picks)]
    by_key = {key: (label, scale, digits) for key, label, scale, digits, _ in METRICS}
    charts = "".join(factor_chart(variants, key, *by_key[key]) for key in
                     ("sam_seabed_0_30_gt_0.5", "sam_seabed_30_60_gt_0.5", "sam_water_gt_0.5", "best_psnr"))
    ordered = [args.baseline] + picks + [n for n in variants if n not in picks and n != args.baseline]
    rule = (f"Najbardziej obiecujące: woda zajęta ≤ {100 * MAX_WATER:g}% i best PSNR ≥ baza − {PSNR_SLACK:g} dB, "
            f"posortowane po średnim pokryciu dna 0–30 i 30–60 px od wody (średnie z seedów).")
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in args.note)
    page = TEMPLATE.format(
        title="Dno i granica z wodą", sources=html.escape(", ".join(str(s) for s in args.sweep)),
        tiles=tiles(variants, args.baseline, picks[0]) if picks else "<p>Żaden wariant nie spełnia reguły.</p>",
        winner=html.escape(short_label(picks[0], args.baseline)) if picks else "–", rule=html.escape(rule), notes=notes,
        baseline=html.escape(base.describe(args.baseline)),
        charts=charts, picks=metric_table(variants, [args.baseline] + picks, args.baseline, picks),
        visuals=visuals(shown, dataset), masks=sam_masks(dataset),
        table=metric_table(variants, ordered, args.baseline, picks))
    args.output.write_text(page)
    print(args.output)


TEMPLATE = """<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
:root{{--bg:#fcfcfb;--panel:#ffffff;--text:#0b0b0b;--muted:#52514e;--line:#e4e3df;--grid:#ecebe7;
--s1:#2a78d6;--s2:#eb6834;--up:#0f7a4f;--down:#b3402a;--pick:#eef4fc}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--panel:#222220;--text:#f0efec;
--muted:#c3c2b7;--line:#3a3936;--grid:#2e2d2b;--s1:#3987e5;--s2:#d95926;--up:#4cc38a;--down:#f08060;--pick:#1f2a38}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--panel:#222220;--text:#f0efec;--muted:#c3c2b7;--line:#3a3936;--grid:#2e2d2b;
--s1:#3987e5;--s2:#d95926;--up:#4cc38a;--down:#f08060;--pick:#1f2a38}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1240px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:1.6rem;margin:0 0 4px}} h2{{font-size:1.2rem;margin:36px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}}
.small{{font-size:.8rem;color:var(--muted)}} .lead{{color:var(--muted);max-width:900px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.tile{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}}
.hero{{font-size:1.6rem;font-weight:600;font-variant-numeric:tabular-nums}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,500px),1fr));gap:16px}}
.viz{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}}
figcaption{{font-weight:600;margin-bottom:4px}} .chart{{width:100%;height:auto}}
.grid{{stroke:var(--grid);stroke-width:1}} .tick{{fill:var(--muted);font-size:11px}} .tick.small{{font-size:9.5px}}
.bar.s1,.dot.s1{{fill:var(--s1)}} .bar.s2,.dot.s2{{fill:var(--s2)}} .dot{{stroke:var(--panel);stroke-width:2}}
.range{{stroke-width:2}} .range.s1{{stroke:var(--s1)}} .range.s2{{stroke:var(--s2)}} .seed{{fill:var(--text);stroke:var(--panel);stroke-width:1.5}}
.hit{{fill:transparent}} .mark:hover .bar{{opacity:.8}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0;font-size:.85rem}}
.key i{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px;border:1px solid var(--line)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:.85rem}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}} td.up{{color:var(--up);font-weight:600}}
td.down{{color:var(--down)}} tr.pick td{{background:var(--pick)}} tr.base td{{font-style:italic}}
.images img{{width:100%;min-width:220px;border-radius:4px;display:block}} .images th{{min-width:120px}}
.tabs{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.tab{{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:6px;padding:4px 10px;cursor:pointer}}
.tab[aria-pressed="true"]{{border-color:var(--s1);color:var(--s1);font-weight:600}}
.masks{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:12px}}
.mask{{margin:0}} .mask img{{width:100%;border-radius:4px}}
</style></head><body><main>
<h1>{title}: co poprawia pokrycie dna i granicę z wodą</h1>
<p class="lead">Szukaliśmy wariantu, w którym zamglone dno przy wodzie jest pokryte Gaussami, woda zostaje czysta,
a PSNR się nie pogarsza. Wszystkie metryki wody i dna liczone na maskach SAM obrazów testowych (sekcja 4), z map α
Gaussów przy najlepszej iteracji; wartości to średnie z seedów (najedź na komórkę, żeby zobaczyć seedy).</p>
<h2>1. Wynik</h2>
<p><b>Najlepszy wariant:</b> {winner}. Duże liczby: ten wariant; pod spodem: baza.</p>
<p class="small">Baza: {baseline}.</p>
<div class="tiles">{tiles}</div>
<ul class="small">{notes}</ul>
<h2>2. Co na co wpływa</h2>
<p class="small">Słupki: średnia z seedów, kropki: pojedyncze seedy. Kolor: <span style="color:var(--s1)">■</span> bez
kompensacji densyfikacji, <span style="color:var(--s2)">■</span> z kompensacją tłumieniem wody. Oś X: waga kary barwy
(chroma) i promienie, na których działa.</p>
<div class="charts">{charts}</div>
<h2>3. Najbardziej obiecujące runy</h2>
<p class="small">{rule} Zielone/czerwone: lepiej/gorzej niż baza.</p>
{picks}
<h3>Wizualizacje (seed 42)</h3>
<p class="small">Mapa pokrycia: szare tło to GT; kolorem nałożone, które piksele dna mają Gaussy (α&gt;0,5), które są
puste (te robi medium) i gdzie Gaussy weszły na wodę.</p>
{visuals}
<h2>4. Maski SAM (woda na czerwono, punkty-podpowiedzi: zielone woda, niebieskie dno)</h2>
<div class="masks">{masks}</div>
<h2>5. Wszystkie warianty</h2>
{table}
<p class="small">Źródła: {sources}</p>
</main>
<script>
document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {{
  document.querySelectorAll('.tab').forEach(t => t.setAttribute('aria-pressed', String(t === tab)));
  document.querySelectorAll('.panel').forEach(p => p.hidden = p.dataset.view !== tab.dataset.view);
}}));
</script></body></html>
"""


if __name__ == "__main__":
    main()
