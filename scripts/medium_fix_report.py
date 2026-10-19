#!/usr/bin/env python3
"""Self-contained HTML report for a medium-fix sweep.

Usage: python scripts/medium_fix_report.py --sweep outputs/iui3_medium_fix/<job> [--sweep <seed job> ...]
       [--output report.html]

Each run directory must contain joint/metrics/eval.csv; audit/best.json and
audit/last.json (scripts/medium_audit.py) are used when present.
"""
import argparse
import base64
import csv
import html
import io
import json
import math
import re
from pathlib import Path

from PIL import Image

VIEW_NAMES = ("MTN_5894", "MTN_5903", "MTN_5911", "MTN_5928")
IMAGE_KINDS = (("hybrid", "hybrid"), ("backscatter", "backscatter (≈ rgb_medium WS)"),
               ("voronoi_only", "voronoi_only"), ("gaussian_only_black", "Gaussy na czarnym"))
DESCRIPTIONS = {
    "B0": "baseline both_sigmoid (kolor per komórka), ten sam kod",
    "E1a_lr1_s01": "globalne c; lr medium 1×; wygładzanie σ 0,1",
    "E1a_lr1_s1": "globalne c; lr medium 1×; wygładzanie σ 1,0",
    "E1a_lr03_s01": "globalne c; lr gęstości/sites 0,3×; wygładzanie σ 0,1",
    "E1a_lr03_s1": "globalne c; lr gęstości/sites 0,3×; wygładzanie σ 1,0",
    "E1b_w001": "E1a + culling 0,5 + pokrycie w=0,01",
    "E1b_w005": "E1a + culling 0,5 + pokrycie w=0,05",
    "E1b_w02": "E1a + culling 0,5 + pokrycie w=0,2",
    "E1b_w001_reset": "E1b w=0,01 + reset opacity→0,5 co 3k, zamrożenie medium 200",
    "E1b_w005_reset": "E1b w=0,05 + reset opacity→0,5 co 3k, zamrożenie medium 200",
    "E1b_w02_reset": "E1b w=0,2 + reset opacity→0,5 co 3k, zamrożenie medium 200",
    "B0_box5k": "B0 + równomierna siatka Voronoi w pudełku sceny, 5000 komórek",
    "B0_box20k": "B0 + równomierna siatka Voronoi w pudełku sceny, 20000 komórek",
    "E1a_lr1_s1_box5k": "E1a_lr1_s1 + równomierna siatka Voronoi, 5000 komórek",
    "E1a_lr1_s1_box20k": "E1a_lr1_s1 + równomierna siatka Voronoi, 20000 komórek",
}
CURVE_FROM = 2000  # early iterations span ~15 dB and would flatten the comparison
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")


LAYOUTS = (("", "normalny"), ("_in75r", "75% pudełko + 25% radialnie"), ("_in50r", "50% pudełko + 50% radialnie"),
           ("_in25r", "25% pudełko + 75% radialnie"), ("_in50n", "50% pudełko + 50% normalnie (odrzucenie)"))
PROX = (("", "1 (brak)"), ("_prox50", "0,5"), ("_prox30", "0,3"), ("_prox10", "0,1"), ("_prox00", "0 (zamrożone)"),
        ("_prox10_ring", "0,1 + sąsiedzi"))


def parse_variant(variant):
    """Split <base>[_in..][_chroma..][_chthr..][_csmooth..][seabed][_cthr..][_send..k][_g..k][_far..][_tden]
    [_water..][_wstart..k][_prox..][_ring][_bk] into (base, layout, prox, black, regularizer)."""
    match = re.fullmatch(r"(.+?)(_in\d\d[rn])?(_chroma\d{3,4})?(_chthr\d\d)?(_csmooth\d+)?"
                         r"(_kappa\d+|_scov\de\d|_scov\d{3}|_under\d{3}|_scale0dens2)?(_cthr\d\d)?(_send\d+k)?"
                         r"(_g\d+k)?(_far\d+s\d{3})?(_tden)?(_water\de\d)?(_wstart\d+k)?(_prox\d\d)?(_ring)?(_bk)?", variant)
    (base, layout, chroma, chroma_threshold, smooth, seabed, threshold, end, gauss, far, compensation, water,
     water_start, prox, ring, black) = match.groups()
    return (base, layout or "", (prox or "") + (ring or ""), bool(black),
            "".join(part or "" for part in (chroma, chroma_threshold, smooth, seabed, threshold, end, gauss, far,
                                            compensation, water, water_start)))


def regularizer_label(code):
    parts = []
    for digit, exponent in re.findall(r"_scov(\d)e(\d)", code):
        parts.append(f"jednostronne pokrycie dna {digit}e-{exponent}")
    code = re.sub(r"_scov\de\d", "", code)
    for cap in re.findall(r"_g(\d+)k", code):
        parts.append(f"do {cap} tys. Gaussów (próg zagęszczania 2e-4)")
    code = re.sub(r"_g\d+k", "", code)
    for threshold in re.findall(r"_chthr(\d\d)", code):
        parts.append(f"kara barwy na promieniach z α > 0,{threshold}")
    if "_tden" in code:
        parts.append("densyfikacja kompensowana tłumieniem wody")
    code = re.sub(r"_chthr\d\d|_tden", "", code)
    for threshold in re.findall(r"_cthr(\d\d)", code):
        parts.append(f"pokrycie dna tylko dla α > 0,{threshold}")
    for end in re.findall(r"_send(\d+)k", code):
        parts.append(f"pokrycie dna wyłączone od {end}k")
    start = re.search(r"_wstart(\d+)k", code)
    for digit, exponent in re.findall(r"_water(\d)e(\d)", code):
        begin = int(start.group(1)) if start else 4
        parts.append(f"strata wodna {digit}e-{exponent} (α < 0,3 → 0, od {begin}k, pełna od {begin + 2}k)")
    code = re.sub(r"_cthr\d\d|_send\d+k|_water\de\d|_wstart\d+k", "", code)
    for distance, scale in re.findall(r"_far(\d+)s(\d{3})", code):
        parts.append(f"usuwanie Gaussów o skali > {fmt(int(scale) / 1000, 2)} dalej niż {distance} j. od kamer")
    code = re.sub(r"_far\d+s\d{3}", "", code)
    for kind, value in re.findall(r"_(chroma|csmooth|kappa|scov|under|scale0dens)(\d+)", code):
        if kind == "chroma":
            parts.append("kara barwy backscatteru " + fmt(int(value) / 100, 2))
        elif kind == "csmooth":
            parts.append("wygładzanie koloru komórek " + value)
        elif kind == "kappa":
            parts.append("ostrość brzegu Gaussów na starcie " + value)
        elif kind == "scov":
            parts.append("jednostronne pokrycie dna " + fmt(int(value) / 100, 2))
        elif kind == "under":
            parts.append("kara za medium pod dnem " + fmt(int(value) / 100, 2))
        else:
            parts.append("bez kary za duże Gaussy + próg zagęszczania 2e-4")
    return ", ".join(parts)


def describe(variant):
    if variant in DESCRIPTIONS:
        return DESCRIPTIONS[variant]
    base, layout, prox, black, regularizer = parse_variant(variant)
    parts = [DESCRIPTIONS.get(base, base)]
    if regularizer:
        parts.append(regularizer_label(regularizer))
    if layout:
        parts.append("komórki: " + dict(LAYOUTS)[layout])
    if prox:
        parts.append("m przy Gaussach = " + dict(PROX).get(prox, prox))
    if black:
        parts.append("czarne tło")
    return "; ".join(parts)


SEABED_CODES = ("_kappa", "_scov", "_under", "_scale0dens2")


FAR_BASE = "B0_chroma100_scov3e3_g700k_bk"
WATER_BASE = "B0_chroma100_g700k_bk"
WATER_CODES = ("_water", "_cthr", "_send")
CHROMA_BASE = "B0_chroma100_scov3e3_cthr90_g700k_water1e2_bk"
CHROMA_CODES = ("_chthr", "_tden")


def baseline_name(variant):
    """Far-pruning variants vs FAR_BASE, seabed variants vs the default
    B0_chroma100_bk, other black-background runs vs B0_bk, the rest vs B0."""
    if any(code in variant for code in CHROMA_CODES):
        return CHROMA_BASE
    if any(code in variant for code in WATER_CODES):
        return WATER_BASE
    if "_far" in variant:
        return FAR_BASE
    if variant.startswith("B0_chroma100_") and any(code in variant for code in SEABED_CODES):
        return "B0_chroma100_bk"
    return "B0_bk" if parse_variant(variant)[3] else "B0"


def family_order(run):
    base, layout, prox, black, regularizer = parse_variant(run["variant"])
    known = list(DESCRIPTIONS)
    return (black, base != "B0", known.index(run["variant"]) if run["variant"] in known else 99, regularizer,
            [l for l, _ in LAYOUTS].index(layout) if layout in dict(LAYOUTS) else 9,
            [p for p, _ in PROX].index(prox) if prox in dict(PROX) else 9, run["seed"] or "")


def read_csv(path):
    with open(path) as file:
        return list(csv.DictReader(file))


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def fmt(value, digits=2, signed=False, percent=False):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if percent:
        value *= 100
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    text = text.replace(".", ",").replace("-", "−")
    return text + ("%" if percent else "")


def load_run(path):
    run = dict(name=path.name, path=path)
    base = path.name.split("_seed")[0]
    run["variant"] = base
    run["seed"] = path.name.split("_seed")[1] if "_seed" in path.name else None
    rows = read_csv(path / "joint/metrics/eval.csv")
    curve = [(int(number(r["iteration"])), number(r["eval/hybrid_psnr"])) for r in rows
             if not math.isnan(number(r.get("eval/hybrid_psnr")))]
    run["curve"] = curve
    if curve:
        best = max(curve, key=lambda item: item[1])
        run["best_iteration"], run["best_psnr"] = best
        run["final_iteration"], run["final_psnr"] = curve[-1]
        best_row = next((r for r in rows if int(number(r["iteration"])) == run["best_iteration"]), {})
        fps = number(best_row.get("eval/hybrid_fps"))
        run["best_fps"] = None if math.isnan(fps) else fps
    for checkpoint in ("best", "last"):
        audit = path / "audit" / f"{checkpoint}.json"
        run[checkpoint] = json.loads(audit.read_text()) if audit.is_file() else None
    run["complete"] = (path / "joint_done.json").is_file()
    run["mask"] = mask_coverage(run)
    return run


WATER_BAND = 30  # pixels on each side of the open-water mask boundary
# Open water vs seabed of the test views, segmented from the images (scripts/sam_water_mask.py).
WATER_MASK = Path(__file__).resolve().parents[1] / "configs/iui3_water_sam.npz"
_REGIONS = {}


def water_regions():
    """Per test view: water, seabed, water within WATER_BAND px of the seabed, seabed within it."""
    if not _REGIONS:
        import numpy as np
        from scipy.ndimage import distance_transform_edt
        with np.load(WATER_MASK) as data:
            for view in (name for name in data.files if name != "meta"):
                water = data[view]
                _REGIONS[view] = dict(water=water, seabed=~water,
                                      band=water & (distance_transform_edt(water) <= WATER_BAND),
                                      seabed_band=~water & (distance_transform_edt(~water) <= WATER_BAND))
    return _REGIONS


def water_history(run, every=1000):
    """Gaussian coverage of open water, of the water band and of the seabed band at every
    ``every``-th evaluation, from the saved Gaussian-only alpha maps; cached in the run's audit/."""
    import numpy as np
    cache = run["path"] / "audit" / f"water_history_{every}_sam.json"
    folders = sorted(path for path in (run["path"] / "joint/evaluation").glob("iteration_*")
                     if int(path.name[-6:]) % every == 0)
    if cache.is_file():
        saved = json.loads(cache.read_text())
        if [row["iteration"] for row in saved] == [int(path.name[-6:]) for path in folders]:
            return saved
    rows = []
    for folder in folders:
        values = {key: [] for key in ("water", "band", "seabed_band")}
        for path in folder.glob("*_gaussian_alpha.npz"):
            regions = water_regions().get(path.name.split("_", 1)[1].split("_gaussian")[0])
            if regions is None:
                continue
            alpha = np.load(path)["alpha"][..., 0]
            values["water"].append(float((alpha[regions["water"]] > 0.5).mean()))
            values["band"].append(float((alpha[regions["band"]] > 0.5).mean()))
            values["seabed_band"].append(float((alpha[regions["seabed_band"]] > 0.99).mean()))
        if values["water"]:
            rows.append(dict(iteration=int(folder.name[-6:]), **{k: float(np.mean(v)) for k, v in values.items()}))
    if rows:
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(rows))
    return rows


def alpha_coverage(folder):
    """Gaussian coverage of the SAM water and seabed regions in one evaluation folder (mean of views)."""
    import numpy as np
    values = {key: [] for key in ("sam_water_gt_0.5", "sam_water_mean", "sam_seabed_gt_0.99", "sam_seabed_gt_0.5",
                                  "sam_band_water_gt_0.5", "sam_band_seabed_gt_0.99")}
    for path in folder.glob("*_gaussian_alpha.npz"):
        regions = water_regions().get(path.name.split("_", 1)[1].split("_gaussian")[0])
        if regions is None:
            continue
        alpha = np.load(path)["alpha"][..., 0]
        values["sam_water_gt_0.5"].append(float((alpha[regions["water"]] > 0.5).mean()))
        values["sam_water_mean"].append(float(alpha[regions["water"]].mean()))
        values["sam_seabed_gt_0.99"].append(float((alpha[regions["seabed"]] > 0.99).mean()))
        values["sam_seabed_gt_0.5"].append(float((alpha[regions["seabed"]] > 0.5).mean()))
        values["sam_band_water_gt_0.5"].append(float((alpha[regions["band"]] > 0.5).mean()))
        values["sam_band_seabed_gt_0.99"].append(float((alpha[regions["seabed_band"]] > 0.99).mean()))
    return {key: float(np.mean(v)) for key, v in values.items() if v}


def mask_coverage(run):
    """alpha_coverage at the best iteration (empty when its alpha maps were not kept); cached."""
    if "best_iteration" not in run:
        return {}
    folder = run["path"] / "joint/evaluation" / f"iteration_{run['best_iteration']:06d}"
    cache = run["path"] / "audit" / f"mask_coverage_sam_{run['best_iteration']:06d}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    result = alpha_coverage(folder) if folder.is_dir() else {}
    if result:
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(result))
    return result


def water_history_charts(runs):
    """Line charts of the water history for the water-clearing comparison (needs >= 3 evaluations)."""
    members = [r for r in runs if r["variant"] == WATER_BASE or baseline_name(r["variant"]) == WATER_BASE]
    histories = [(r, water_history(r)) for r in members]
    histories = [(r, h) for r, h in histories if len(h) >= 3]
    if len(histories) < 2:
        return ""
    charts = []
    for key, title in (("water", "Otwarta woda zajęta przez Gaussy (α > 0,5) [%]"),
                       ("band", f"Woda w pasie {WATER_BAND} px przy dnie zajęta przez Gaussy (α > 0,5) [%]"),
                       ("seabed_band", f"Dno w pasie {WATER_BAND} px przy wodzie pokryte (α > 0,99) [%]")):
        series = [dict(r, curve=[(row["iteration"], 100 * row[key]) for row in h]) for r, h in histories]
        charts.append(line_chart(f"water_{key}", f"{title}, w trakcie treningu", series, curve_from=0,
                                 unit="%", digits=1))
    return ("<h3>Woda w trakcie treningu</h3><p class='small'>Z map α Gaussów zapisywanych przy ewaluacji co "
            "1000 iteracji; maska otwartej wody jak w audycie. Do 3–4k Gaussy z inicjalizacji pokrywają prawie "
            "cały kadr, czyści je rampa progu opacity.</p>" + "".join(charts))


def audit_value(run, key, checkpoint="best"):
    if checkpoint == "best" and key in run.get("mask", {}):
        return run["mask"][key]
    audit = run.get(checkpoint)
    if not audit:
        return float("nan")
    return audit["mean"].get(key, float("nan"))


def thumbnail(path, width=300):
    if not path.is_file():
        return None
    image = Image.open(path).convert("RGB")
    image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def select_winner(runs, baseline_variant="B0"):
    """Best hybrid PSNR among runs that lose <=0.5 dB to the baseline and disentangle clearly better."""
    runs = [r for r in runs if baseline_name(r["variant"]) == baseline_variant or r["variant"] == baseline_variant]
    baseline = next((r for r in runs if r["variant"] == baseline_variant and r["seed"] is None), None)
    if baseline is None or "best_psnr" not in baseline or not baseline.get("best"):
        return None, f"Brak kompletnego {baseline_variant} – nie da się zastosować reguły wyboru."
    b_drop = abs(audit_value(baseline, "clean_near/delta_psnr_bottom"))
    b_cover = seabed_cover(baseline)
    eligible = []
    for run in runs:
        if run is baseline or run["seed"] is not None or "best_psnr" not in run or not run.get("best"):
            continue
        drop = abs(audit_value(run, "clean_near/delta_psnr_bottom"))
        cover = seabed_cover(run)
        if (run["best_psnr"] >= baseline["best_psnr"] - 0.5 and drop <= 0.5 * b_drop and cover >= b_cover):
            eligible.append(run)
    b = baseline_variant
    rule = (f"Reguła ({b}): best PSNR ≥ {b} − 0,5 dB ({fmt(baseline['best_psnr'] - 0.5)} dB), "
            f"|ΔPSNR dna po czyszczeniu komórek ≤5 j.| ≤ połowa {b} ({fmt(0.5 * b_drop)} dB) "
            f"oraz pokrycie dna α>0,99 ≥ {b} ({fmt(b_cover, 1, percent=True)}).")
    if not eligible:
        return None, rule + " Żaden run nie spełnia wszystkich warunków."
    winner = max(eligible, key=lambda r: r["best_psnr"])
    names = ", ".join(r["name"] for r in sorted(eligible, key=lambda r: -r["best_psnr"]))
    return winner, rule + f" Spełniają: {names}."


def nice_step(span, ticks=6):
    """Grid step of 1, 2 or 5 x 10^k giving at most about ``ticks`` intervals."""
    raw = max(span, 1e-9) / ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    return next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)


def line_chart(chart_id, title, runs, curve_from=CURVE_FROM, unit="dB", digits=3):
    width, height, left, right, top, bottom = 760, 300, 52, 16, 16, 34
    series = [(dict(run, curve=[p for p in run["curve"] if p[0] >= curve_from]), i + 1)
              for i, run in enumerate(runs) if run["curve"]][:len(SERIES)]
    if not series:
        return ""
    xs = [x for run, _ in series for x, _ in run["curve"]]
    ys = [y for run, _ in series for _, y in run["curve"]]
    x0, x1 = min(xs), max(xs)
    tick = nice_step(max(ys) - min(ys))
    y0, y1 = math.floor(min(ys) / tick) * tick, math.ceil(max(ys) / tick) * tick
    y1 = y1 if y1 > y0 else y0 + tick
    sx = lambda x: left + (x - x0) / max(x1 - x0, 1) * (width - left - right)  # noqa: E731
    sy = lambda y: top + (y1 - min(max(y, y0), y1)) / max(y1 - y0, 1e-6) * (height - top - bottom)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(title)}">']
    y = y0
    while y <= y1 + 1e-9:
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{sy(y):.1f}" y2="{sy(y):.1f}"/>'
                     f'<text class="tick" x="{left - 6}" y="{sy(y) + 4:.1f}" text-anchor="end">{fmt(y, 1)}</text>')
        y += tick
    for x in range(math.ceil(x0 / 5000) * 5000, x1 + 1, 5000):
        parts.append(f'<text class="tick" x="{sx(x):.1f}" y="{height - 12}" text-anchor="middle">{x // 1000}k</text>')
    parts.append(f'<text class="tick" x="{width - right}" y="{height - 0}" text-anchor="end">iteracja</text>')
    for run, slot in series:
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in run["curve"])
        parts.append(f'<polyline class="series s{slot}" points="{points}"/>')
    parts.append(f'<line class="crosshair" x1="0" x2="0" y1="{top}" y2="{height - bottom}" visibility="hidden"/>')
    parts.append(f'<rect class="hit" x="{left}" y="{top}" width="{width - left - right}" '
                 f'height="{height - top - bottom}" fill="transparent"/></svg>')
    data = dict(left=left, right=right, width=width, x0=x0, x1=x1, unit=unit, digits=digits,
                series=[dict(name=run["name"], slot=slot, curve=run["curve"]) for run, slot in series])
    legend = "".join(f'<span class="key"><i class="sw s{slot}"></i>{html.escape(run["name"])}</span>'
                     for run, slot in series)
    return (f'<figure class="viz" id="{chart_id}" data-chart=\'{html.escape(json.dumps(data))}\'>'
            f'<figcaption>{html.escape(title)}</figcaption><div class="legend">{legend}</div>'
            f'{"".join(parts)}<div class="tooltip" hidden></div></figure>')


def bar_chart(title, rows, digits=2, percent=False, note=""):
    """Horizontal single-series bars; rows = [(label, value)]. Values may be negative."""
    rows = [(label, value) for label, value in rows if not math.isnan(value)]
    if not rows:
        return ""
    label_width, width, bar_h, gap = 150, 760, 18, 8
    height = len(rows) * (bar_h + gap) + 8
    low, high = min(0.0, min(v for _, v in rows)), max(0.0, max(v for _, v in rows))
    span = (high - low) or 1.0
    plot = width - label_width - 70
    sx = lambda v: label_width + (v - low) / span * plot  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(title)}">']
    zero = sx(0.0)
    for index, (label, value) in enumerate(rows):
        y = 4 + index * (bar_h + gap)
        start, end = sorted((zero, sx(value)))
        length = max(end - start, 1.0)
        r = min(4.0, length / 2)
        if value >= 0:  # rounded data end on the right, square at the baseline
            path = (f"M{start:.1f},{y} H{start + length - r:.1f} Q{start + length:.1f},{y} {start + length:.1f},{y + r} "
                    f"V{y + bar_h - r} Q{start + length:.1f},{y + bar_h} {start + length - r:.1f},{y + bar_h} H{start:.1f} Z")
            text_x, anchor = start + length + 6, "start"
        else:
            path = (f"M{end:.1f},{y} H{start + r:.1f} Q{start:.1f},{y} {start:.1f},{y + r} V{y + bar_h - r} "
                    f"Q{start:.1f},{y + bar_h} {start + r:.1f},{y + bar_h} H{end:.1f} Z")
            text_x, anchor = zero + 6, "start"
        shown = fmt(value, digits, percent=percent, signed=low < 0)
        parts.append(f'<text class="label" x="{label_width - 8}" y="{y + 13}" text-anchor="end">{html.escape(label)}</text>'
                     f'<path class="bar" d="{path}"><title>{html.escape(label)}: {shown}</title></path>'
                     f'<text class="value" x="{text_x:.1f}" y="{y + 13}" text-anchor="{anchor}">{shown}</text>')
    parts.append(f'<line class="axis" x1="{zero:.1f}" x2="{zero:.1f}" y1="0" y2="{height}"/></svg>')
    note_html = f'<p class="small">{note}</p>' if note else ""
    return f'<figure class="viz"><figcaption>{html.escape(title)}</figcaption>{"".join(parts)}{note_html}</figure>'


def image_grid(runs):
    tabs, panels = [], []
    for view_index, view in enumerate(VIEW_NAMES):
        tabs.append(f'<button class="tab" data-view="{view_index}" aria-pressed="{str(view_index == 2).lower()}">{view}</button>')
        rows = []
        for run in runs:
            if "best_iteration" not in run:
                continue
            folder = run["path"] / "joint/evaluation" / f"iteration_{run['best_iteration']:06d}"
            cells = []
            for kind, _ in IMAGE_KINDS:
                source = thumbnail(folder / f"{view_index:04d}_{view}_{kind}.png")
                cells.append(f'<td><img loading="lazy" src="{source}" alt="{kind}"></td>' if source else "<td>–</td>")
            rows.append(f'<tr><th>{html.escape(run["name"])}<br><span class="small">it. {run["best_iteration"]}</span></th>'
                        f'{"".join(cells)}</tr>')
        header = "".join(f"<th>{label}</th>" for _, label in IMAGE_KINDS)
        panels.append(f'<div class="panel" data-view="{view_index}" {"" if view_index == 2 else "hidden"}>'
                      f'<div class="tw"><table class="images"><tr><th></th>{header}</tr>{"".join(rows)}</table></div></div>')
    return f'<div class="tabs">{"".join(tabs)}</div>{"".join(panels)}'


def matrix(title, rows, columns, lookup):
    """rows/columns: [(key, label)]; lookup(row, column) -> run or None."""
    header = "".join(f"<th class='num'>{html.escape(label)}</th>" for _, label in columns)
    body, filled = [], False
    for row, row_label in rows:
        cells = []
        for column, _ in columns:
            run = lookup(row, column)
            if run is None or "best_psnr" not in run:
                cells.append("<td class='num'>–</td>")
                continue
            filled = True
            cells.append(
                f"<td class='num' title='{html.escape(run['name'])}'><b>{fmt(run['best_psnr'], 2)}</b><br>"
                f"<span class='small'>woda {fmt(audit_value(run, 'original/psnr_water'), 1)} · "
                f"ΔPSNR dna {fmt(audit_value(run, 'clean_near/delta_psnr_bottom'), 1, signed=True)}<br>"
                f"dno α&gt;0,99 {fmt(seabed_cover(run), 0, percent=True)} · "
                f"woda zajęta (α&gt;0,5) {fmt(audit_value(run, 'sam_water_gt_0.5'), 1, percent=True)}"
                f"{hue_text(run)}</span></td>")
        body.append(f"<tr><th>{html.escape(row_label)}</th>{''.join(cells)}</tr>")
    if not filled:
        return ""
    return (f"<h3>{html.escape(title)}</h3><div class='tw'><table><tr><th></th>{header}</tr>{''.join(body)}</table></div>")


def hue_text(run):
    angle = audit_value(run, "backscatter_hue_angle_bottom")
    text = "" if math.isnan(angle) else f"<br>kąt barwy backscatteru dna {fmt(angle, 1)}°"
    partial = audit_value(run, "sam_seabed_gt_0.5") - audit_value(run, "sam_seabed_gt_0.99")
    text += "" if math.isnan(partial) else f"<br>dno półprzezroczyste (α 0,5–0,99) {fmt(partial, 0, percent=True)}"
    band_water, band_seabed = audit_value(run, "sam_band_water_gt_0.5"), audit_value(run, "sam_band_seabed_gt_0.99")
    if not math.isnan(band_water):
        text += (f"<br>pas {WATER_BAND} px przy granicy: woda zajęta {fmt(band_water, 0, percent=True)} · "
                 f"dno pokryte {fmt(band_seabed, 0, percent=True)}")
    count = ((run.get("best") or {}).get("gaussians") or {}).get("count")
    fps = run.get("best_fps")
    if count:
        text += f"<br>Gaussów {count / 1000:.0f} tys."
    if fps:
        text += f" · {fps:.0f} FPS"
    return text


def matrices(runs, all_runs=()):
    by_name = {r["variant"]: r for r in runs}
    bases = (("B0", "B0 (kolor per komórka)"), ("E1a_lr1_s1", "lider (globalne c)"))
    stage_a = matrix("Etap A (czarne tło): rozkład komórek × tryb koloru", LAYOUTS, bases,
                     lambda layout, base: by_name.get(f"{base}{layout}_bk"))
    columns = [((base, layout), f"{label}, {name}") for base, label in bases
               for layout, name in (("", "normalny"), ("_in50r", "50/50 radialnie"))]
    prox = matrix("Czarne tło: m (krok medium w komórkach z Gaussami) × tryb koloru × rozkład", PROX, columns,
                  lambda p, key: by_name.get(f"{key[0]}{key[1]}{p}_bk"))
    by_full = {r["name"]: r for r in all_runs}
    settings = [("", "B0_bk (bez dodatków)")] + [(code, regularizer_label(code)) for code in
                ("_chroma001", "_chroma010", "_chroma100", "_csmooth10", "_csmooth100")]
    chroma = matrix("Czarne tło, B0: kara barwy backscatteru i wygładzanie koloru komórek × seed", settings,
                    (("", "seed 42"), ("_seed43", "seed 43")),
                    lambda code, seed: by_full.get(f"B0{code}_bk{seed}"))
    seabed_rows = [("", "domyślna B0_chroma100_bk")] + [(code, regularizer_label(code)) for code in
                   ("_kappa2", "_kappa4", "_scov1e4", "_scov1e3", "_scov3e3", "_scov001", "_scov010", "_scov100", "_scov200",
                    "_under001", "_under010", "_under100", "_under200", "_scale0dens2")]
    seabed = matrix("Czarne tło, domyślna konfiguracja: pokrycie dna Gaussami × seed", seabed_rows,
                    (("", "seed 42"), ("_seed43", "seed 43")),
                    lambda code, seed: by_full.get(f"B0_chroma100{code}_bk{seed}"))
    bases = (("B0", "B0_bk"), ("B0_chroma100", "domyślna B0_chroma100_bk"),
             ("B0_chroma100_scov3e3", "domyślna + pokrycie dna 3e-3"))
    caps = (("", "ok. 290 tys. (próg 4e-4)"), ("500", "do 500 tys."), ("700", "do 700 tys."), ("1000", "do 1 mln"))
    gauss = "".join(
        matrix(f"Czarne tło: baza × liczba Gaussów, seed {label}", bases, caps,
               lambda base, cap, seed=seed: by_full.get(f"{base}_bk{seed}" if not cap else f"{base}_g{cap}k_bk{seed}"))
        for seed, label in (("", "42"), ("_seed43", "43")))
    distances = (("", "bez usuwania"), ("10", "odległość > 10"), ("15", "odległość > 15"), ("20", "odległość > 20"))
    scales = (("030", "skala > 0,03"), ("050", "skala > 0,05"), ("080", "skala > 0,08"))
    base = FAR_BASE.removesuffix("_g700k_bk")
    far = "".join(
        matrix(f"Domyślna + pokrycie dna 3e-3, do 700 tys. Gaussów: usuwanie dużych Gaussów daleko od kamer, seed {label}",
               distances, scales,
               lambda d, sc, seed=seed: by_full.get(f"{FAR_BASE}{seed}" if not d else f"{base}_g700k_far{d}s{sc}_bk{seed}"))
        for seed, label in (("", "42"), ("_seed43", "43")))
    far_scov = matrix("Usuwanie (odległość > 15, skala > 0,05) × waga pokrycia dna",
                      (("_scov3e3", "pokrycie 3e-3"), ("_scov5e3", "pokrycie 5e-3"), ("_scov1e2", "pokrycie 1e-2")),
                      (("", "seed 42"), ("_seed43", "seed 43")),
                      lambda code, seed: by_full.get(f"B0_chroma100{code}_g700k_far15s050_bk{seed}"))
    water_rows = [("", "bez straty wodnej")] + [(f"_water{code}", f"strata wodna {code.replace('e', 'e-')}")
                                                 for code in ("1e3", "1e2", "1e1", "1e0")]
    water = matrix("Do 700 tys. Gaussów, bez pokrycia dna: waga straty wodnej × start × seed", water_rows,
                   ((("", ""), "od 4k, seed 42"), (("", "_seed43"), "od 4k, seed 43"),
                    (("_wstart13k", ""), "od 13k, seed 42"), (("_wstart13k", "_seed43"), "od 13k, seed 43")),
                   # The control has no water loss, hence no late-start variant.
                   lambda code, column: by_full.get(
                       f"B0_chroma100_g700k{code}{column[0] if code else ''}_bk{column[1]}"))
    water_seabed = matrix("Pokrycie dna 3e-3 bez rozlewania na wodę × seed",
                          (("_scov3e3_cthr90_g700k_water1e2", "pokrycie tylko dla α > 0,9 + strata wodna 1e-2"),
                           ("_scov3e3_send15k_g700k", "pokrycie wyłączone od 15k")),
                          (("", "seed 42"), ("_seed43", "seed 43")),
                          lambda code, seed: by_full.get(f"B0_chroma100{code}_bk{seed}"))
    tail = "_scov3e3_cthr90_g700k"
    chroma_rows = [("", "baza: chroma 1 tylko dla α > 0,5")] + [
        (f"_chroma{code}_chthr02", f"chroma {label} na wszystkich promieniach")
        for code, label in (("010", "0,1"), ("030", "0,3"), ("100", "1"), ("300", "3"), ("1000", "10"))]
    chroma_all = matrix(
        "Pokrycie dna α > 0,9 + woda 1e-2: chroma na wszystkich promieniach × densyfikacja kompensowana × seed",
        chroma_rows, ((("", ""), "bez kompensacji, seed 42"), (("", "_seed43"), "bez kompensacji, seed 43"),
                      (("_tden", ""), "kompensacja, seed 42"), (("_tden", "_seed43"), "kompensacja, seed 43")),
        lambda code, column: by_full.get(f"B0{code or '_chroma100'}{tail}{column[0]}_water1e2_bk{column[1]}"))
    gauss = chroma_all + water + water_seabed + far + far_scov + gauss
    if not (stage_a or prox or chroma or seabed or gauss):
        return ""
    legend = ("<p class='small'>W komórce: <b>best test PSNR</b>; PSNR wody; ΔPSNR dna po zastąpieniu komórek ≤5 j. "
              "czystą wodą (bliżej 0 = medium mniej podpiera dno); pokrycie dna Gaussami; odsetek pikseli wody bez Gaussów "
              "(spadek = Gaussy wchodzą na wodę). Wiersz „1 (brak)” = run z etapu A.</p>")
    return "<h2>Macierze porównań</h2>" + legend + gauss + seabed + chroma + stage_a + prox


def seed_table(runs):
    groups = {}
    for run in runs:
        groups.setdefault(run["variant"], []).append(run)
    rows = []
    for variant, members in groups.items():
        if len(members) < 2:
            continue
        def stats(values):
            values = [v for v in values if not math.isnan(v)]
            if not values:
                return "–"
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5
            return f"{fmt(mean)} ± {fmt(std)} (n={len(values)})"
        rows.append(f"<tr><td>{html.escape(variant)}</td>"
                    f"<td class='num'>{stats([m.get('best_psnr', float('nan')) for m in members])}</td>"
                    f"<td class='num'>{stats([m.get('final_psnr', float('nan')) for m in members])}</td>"
                    f"<td class='num'>{stats([audit_value(m, 'clean_near/delta_psnr_bottom') for m in members])}</td>"
                    f"<td class='num'>{stats([100 * seabed_cover(m) for m in members])}</td>"
                    f"<td class='num'>{stats([100 * audit_value(m, 'sam_water_gt_0.5') for m in members])}</td>"
                    f"<td class='num'>{stats([100 * audit_value(m, 'sam_band_water_gt_0.5') for m in members])}</td>"
                    f"<td class='num'>{stats([100 * audit_value(m, 'sam_band_seabed_gt_0.99') for m in members])}</td>"
                    f"<td class='num'>{stats([gaussian_count(m) / 1000 for m in members])}</td></tr>")
    if not rows:
        return "<p>Brak wariantów z więcej niż jednym seedem.</p>"
    return ("<div class='tw'><table><tr><th>Wariant</th><th class='num'>best PSNR</th><th class='num'>final PSNR</th>"
            "<th class='num'>ΔPSNR dna (czyszczenie ≤5 j.)</th><th class='num'>dno α&gt;0,99 [%]</th>"
            "<th class='num'>woda zajęta, α&gt;0,5 [%]</th><th class='num'>pas: woda zajęta [%]</th>"
            "<th class='num'>pas: dno pokryte [%]</th><th class='num'>Gaussy [tys.]</th></tr>"
            + "".join(rows) + "</table></div>")


def seabed_cover(run):
    """Seabed covered by Gaussians (alpha > 0.99): on the SAM seabed when the best alpha maps
    exist, otherwise the audit's lower 40% of the frame."""
    value = audit_value(run, "sam_seabed_gt_0.99")
    return audit_value(run, "bottom_alpha_gt_0.99") if math.isnan(value) else value


def gaussian_count(run):
    count = ((run.get("best") or {}).get("gaussians") or {}).get("count")
    return float("nan") if count is None else float(count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--note", action="append", default=[], help="extra remark shown in the summary (repeatable)")
    args = parser.parse_args()
    runs = []
    for sweep in args.sweep:
        for path in sorted(sweep.iterdir()):
            if (path / "joint/metrics/eval.csv").is_file():
                runs.append(load_run(path))
    runs.sort(key=family_order)
    main_runs = [r for r in runs if r["seed"] is None]
    winners, rules = [], []
    for baseline in ("B0", "B0_bk", "B0_chroma100_bk", FAR_BASE, WATER_BASE, CHROMA_BASE):
        compared = [r for r in main_runs if baseline_name(r["variant"]) == baseline and r["variant"] != baseline]
        if compared and any(r["variant"] == baseline for r in main_runs):
            found, text = select_winner(main_runs, baseline)
            winners.append((baseline, found))
            rules.append(text)
    winner_set = {id(w) for _, w in winners if w is not None}

    table_rows = []
    for run in runs:
        is_winner = id(run) in winner_set
        global_color = (run["best"] or {}).get("global_color")
        swatch = ("<span class='swatch' style='background:rgb({})'></span>".format(
            ",".join(str(round(255 * min(max(c, 0), 1))) for c in global_color)) if global_color else "")
        table_rows.append(
            f"<tr class='{'winner' if is_winner else ''}'><td>{html.escape(run['name'])}{' ★' if is_winner else ''}"
            f"{'' if run['complete'] else ' <span class=small>(niekompletny)</span>'}</td>"
            f"<td>{html.escape(describe(run['variant']))}</td>"
            f"<td class='num'>{fmt(run.get('best_psnr', float('nan')), 3)}</td>"
            f"<td class='num'>{run.get('best_iteration', '–')}</td>"
            f"<td class='num'>{fmt(run.get('final_psnr', float('nan')), 3)}</td>"
            f"<td class='num'>{fmt(audit_value(run, 'original/psnr_bottom'))}</td>"
            f"<td class='num'>{fmt(audit_value(run, 'gaussian_only_black_psnr_bottom'))}</td>"
            f"<td class='num'>{fmt(seabed_cover(run), 1, percent=True)}</td>"
            f"<td class='num'>{fmt(audit_value(run, 'behind_share_bottom_mean'), 1, percent=True)}</td>"
            f"<td class='num'>{fmt(audit_value(run, 'clean_anomalous/delta_psnr_bottom'), signed=True)}</td>"
            f"<td class='num'>{fmt(audit_value(run, 'clean_near/delta_psnr_bottom'), signed=True)}</td>"
            f"<td class='num'>{swatch}{fmt(audit_value(run, 'marcher_mae_hybrid'), 4)}</td></tr>")

    psnr_of = {r["variant"]: r.get("best_psnr", float("nan")) for r in main_runs}
    white = [r for r in main_runs if not parse_variant(r["variant"])[3]]
    groups = [
        ("B0 i E1a", [r for r in white if r["variant"] == "B0" or (r["variant"].startswith("E1a") and "_box" not in r["variant"])]),
        ("B0 i E1b", [r for r in white if r["variant"] == "B0" or r["variant"].startswith("E1b")]),
        ("rozkład komórek: normalny vs równomierny", [r for r in white if r["variant"] in ("B0", "E1a_lr1_s1") or "_box" in r["variant"]]),
    ]
    for base, label in (("B0", "B0"), ("E1a_lr1_s1", "lider")):
        black_runs = [r for r in main_runs if parse_variant(r["variant"])[3] and parse_variant(r["variant"])[0] == base
                      and not parse_variant(r["variant"])[4]]
        groups.append((f"czarne tło, {label}: rozkład komórek",
                       [r for r in black_runs if not parse_variant(r["variant"])[2]]))
        for layout, layout_label in (("", "rozkład normalny"), ("_in50r", "50% pudełko + 50% radialnie")):
            groups.append((f"czarne tło, {label}, {layout_label}: m przy Gaussach",
                           [r for r in black_runs if parse_variant(r["variant"])[1] == layout]))
    seabed_codes = SEABED_CODES
    groups.append(("domyślna konfiguracja: ostrość brzegu i jednostronne pokrycie dna",
                   [r for r in main_runs if r["variant"] == "B0_chroma100_bk"
                    or any(c in r["variant"] for c in ("_kappa", "_scov", "_scale0dens2"))]))
    groups.append(("domyślna konfiguracja: kara za medium pod dnem",
                   [r for r in main_runs if r["variant"] == "B0_chroma100_bk" or "_under" in r["variant"]]))
    for base, label in (("B0", "B0_bk"), ("B0_chroma100", "domyślna"), ("B0_chroma100_scov3e3", "domyślna + pokrycie 3e-3")):
        groups.append((f"{label}: liczba Gaussów",
                       [r for r in main_runs if r["variant"] == f"{base}_bk" or re.fullmatch(rf"{base}_g\d+k_bk", r["variant"])]))
    for distance in ("10", "15", "20"):
        groups.append((f"usuwanie dużych Gaussów dalej niż {distance} j. od kamer",
                       [r for r in main_runs if r["variant"] == FAR_BASE or f"_far{distance}s" in r["variant"]]))
    groups.append(("czarne tło, B0: kara barwy backscatteru i wygładzanie koloru",
                   [r for r in main_runs if r["variant"] == "B0_bk"
                    or (parse_variant(r["variant"])[4] and parse_variant(r["variant"])[0] == "B0"
                        and not parse_variant(r["variant"])[1]
                        and not any(c in r["variant"] for c in seabed_codes))]))
    groups.append(("strata wodna i pokrycie dna bez rozlewania",
                   [r for r in main_runs if r["variant"] == WATER_BASE or baseline_name(r["variant"]) == WATER_BASE]))
    charts = water_history_charts(main_runs) + "".join(
        line_chart(f"curve_{i}", f"Test hybrid PSNR [dB] od 2k iteracji: {title}", members)
        for i, (title, members) in enumerate(groups)
        if len(members) >= 2 and any(parse_variant(r["variant"])[0] != "B0" or r["variant"] not in ("B0", "B0_bk")
                                     for r in members))
    bars = (
        bar_chart("Best test hybrid PSNR względem bazy (B0; czarne tło: B0_bk; warianty dna: B0_chroma100_bk) [dB]",
                  [(r["name"], r.get("best_psnr", float("nan")) - psnr_of.get(baseline_name(r["variant"]), float("nan")))
                   for r in main_runs if r["variant"] not in ("B0", "B0_bk")], 3,
                  note="Reguła wyboru dopuszcza spadek do −0,5 dB względem swojej bazy.")
        + bar_chart("Otwarta woda zajęta przez Gaussy: udział pikseli maski wody z α > 0,5 (best; im mniej, tym lepiej)",
                    [(r["name"], audit_value(r, "sam_water_gt_0.5")) for r in main_runs], 1, percent=True,
                    note="Maska wody: segmentacja obrazów testowych SAM (configs/iui3_water_sam.npz), α z ewaluacji best.")
        + bar_chart("Pokrycie dna Gaussami: udział pikseli z α > 0,99 (best)",
                    [(r["name"], seabed_cover(r)) for r in main_runs], 1, percent=True,
                    note="Na całym dnie wg maski SAM; runy bez zapisanych map α: dolne 40% kadru z audytu.")
        + bar_chart("Udział backscatteru zza powierzchni na dnie (best; im mniej, tym lepiej)",
                    [(r["name"], audit_value(r, "behind_share_bottom_mean")) for r in main_runs], 1, percent=True)
        + bar_chart("ΔPSNR dna po zastąpieniu komórek ≤5 j. od Gaussów czystą wodą (best; bliżej 0 = lepsze rozdzielenie)",
                    [(r["name"], audit_value(r, "clean_near/delta_psnr_bottom")) for r in main_runs], 2,
                    note="Gaussy nie są douczane po zmianie – to miara zależności rozwiązania od lokalnego medium."))

    winner_html = "".join(
        (f"<p><b>Zwycięzca względem {baseline}: {html.escape(found['name'])}</b> "
         f"(best PSNR {fmt(found['best_psnr'], 3)} dB, it. {found['best_iteration']}).</p>")
        if found else f"<p><b>Brak zwycięzcy względem {baseline}.</b></p>" for baseline, found in winners)
    rule = " ".join(rules)
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in args.note)
    winner_html += f"<ul class='small'>{notes}</ul>" if notes else ""
    template = TEMPLATE.replace("{{RULE}}", html.escape(rule)).replace("{{WINNER}}", winner_html)
    template = template.replace("{{TABLE}}", "".join(table_rows)).replace("{{CHARTS}}", charts)
    template = template.replace("{{MATRIX}}", matrices(main_runs, runs))
    template = template.replace("{{BARS}}", bars).replace("{{IMAGES}}", image_grid(main_runs))
    template = template.replace("{{SEEDS}}", seed_table(runs))
    template = template.replace("{{SOURCES}}", html.escape(", ".join(str(s) for s in args.sweep)))
    slots = "".join(f".s{i + 1}{{--c:{c}}}" for i, c in enumerate(SERIES))
    media = "".join(f':root:not([data-theme="light"]) .s{i + 1}{{--c:{c}}}' for i, c in enumerate(SERIES_DARK))
    toggle = "".join(f':root[data-theme="dark"] .s{i + 1}{{--c:{c}}}' for i, c in enumerate(SERIES_DARK))
    template = template.replace("{{SLOTS}}", slots).replace("{{SLOTS_MEDIA}}", media).replace("{{SLOTS_TOGGLE}}", toggle)
    output = args.output or args.sweep[0] / "report.html"
    output.write_text(template, encoding="utf-8")
    print(output)


TEMPLATE = """<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sweep medium fix</title>
<style>
:root{--bg:#f7f8fa;--panel:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#dfe3e8;--grid:#ebeae6;--accent:#1f5fae;
--accent-soft:#e7effa;--code:#eef1f5;--bar:#2a78d6;--win:#fff6dd}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111418;--panel:#1a1a19;--ink:#f2f2f0;
--muted:#c3c2b7;--line:#2b3138;--grid:#2c2c2a;--accent:#7fb0ef;--accent-soft:#1b2a3d;--code:#20262d;--bar:#3987e5;--win:#33301f}}
:root[data-theme="dark"]{--bg:#111418;--panel:#1a1a19;--ink:#f2f2f0;--muted:#c3c2b7;--line:#2b3138;--grid:#2c2c2a;
--accent:#7fb0ef;--accent-soft:#1b2a3d;--code:#20262d;--bar:#3987e5;--win:#33301f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 80px}h1{font-size:1.7rem;margin:0 0 6px}
h2{font-size:1.25rem;margin:40px 0 10px;padding-top:10px;border-top:1px solid var(--line)}
p,li{max-width:80ch}.small{font-size:.85rem;color:var(--muted)}code{font-family:ui-monospace,Menlo,Consolas,monospace;background:var(--code);padding:1px 5px;border-radius:4px;font-size:.88em}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.tldr{border-left:4px solid var(--accent);background:var(--accent-soft)}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{background:var(--code);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.runs td:nth-child(2){min-width:300px}tr.winner td{background:var(--win)}
.swatch{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px;border:1px solid var(--line)}
.viz{margin:18px 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;position:relative}
.viz figcaption{font-weight:600;margin-bottom:6px}.chart{width:100%;height:auto;display:block}
.chart .grid{stroke:var(--grid);stroke-width:1}.chart .axis{stroke:var(--muted);stroke-width:1}
.chart text{fill:var(--muted);font-size:11px}.chart .label{fill:var(--ink);font-size:12px}.chart .value{fill:var(--ink);font-size:12px;font-variant-numeric:tabular-nums}
{{SLOTS}}@media (prefers-color-scheme: dark){ {{SLOTS_MEDIA}} }
{{SLOTS_TOGGLE}}
.chart .series{fill:none;stroke:var(--c);stroke-width:2;stroke-linejoin:round}.sw{background:var(--c)}.chart .bar{fill:var(--bar)}.chart .bar:hover{opacity:.8}
.chart .crosshair{stroke:var(--muted);stroke-dasharray:3 3}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:.85rem;margin-bottom:6px}.key i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.tooltip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:.8rem;box-shadow:0 2px 8px rgba(0,0,0,.15);white-space:nowrap}
.tooltip i{display:inline-block;width:10px;height:3px;margin-right:5px;vertical-align:middle;background:var(--c)}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}.tab{font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:999px;padding:3px 12px;cursor:pointer}
.tab[aria-pressed="true"]{background:var(--accent-soft);border-color:var(--accent)}
table.images td{padding:3px}table.images img{width:100%;min-width:180px;max-width:300px;display:block;border-radius:4px}
</style></head><body><main>
<h1>Sweep: globalne c, pokrycie dna i culling w stylu WS</h1>
<p class="small">Źródła: {{SOURCES}}. Metryki obrazu: <code>joint/metrics/eval.csv</code> (pełna rozdzielczość, 4 widoki testowe).
Metryki rozdzielenia: <code>scripts/medium_audit.py</code> na checkpoincie best (piksele co 4., „dno” = dolne 40% kadru).</p>
<div class="card tldr"><h2 style="margin-top:0;border:0;padding:0">Wynik</h2>{{WINNER}}<p class="small">{{RULE}}</p></div>
<h2>Tabela runów</h2>
<div class="tw"><table class="runs"><tr><th>Run</th><th>Ustawienie</th><th class="num">best PSNR</th><th class="num">it.</th>
<th class="num">final PSNR</th><th class="num">PSNR dna</th><th class="num">Gaussy/czarne PSNR dna</th>
<th class="num">dno α&gt;0,99</th><th class="num">backscatter zza dna</th><th class="num">ΔPSNR dna: anomalne→woda</th>
<th class="num">ΔPSNR dna: ≤5 j.→woda</th><th class="num">c / MAE marchera</th></tr>{{TABLE}}</table></div>
<p class="small">Kolumny od „PSNR dna” w prawo pochodzą z audytu checkpointu best. „Gaussy/czarne” = render samych Gaussów na czarnym tle względem podwodnego GT (diagnostyka, nie miara fizyczna).
Kolorowy kwadrat = nauczony globalny kolor wody c. MAE marchera = zgodność niezależnej reimplementacji z rendererem (wiarygodność kolumn audytu).</p>
{{MATRIX}}
<h2>Krzywe uczenia</h2>{{CHARTS}}
<h2>Metryki kluczowe</h2>{{BARS}}
<h2>Obrazy (checkpoint best)</h2>
<p class="small"><b>backscatter</b> to odpowiednik obrazu „Medium” z water-splatting. Celem jest gładki gradient wody bez tekstury dna.</p>
{{IMAGES}}
<h2>Seedy</h2>{{SEEDS}}
<h2>Zastrzeżenia</h2><ul>
<li>Jeden seed na ustawienie w sweepie głównym; różnice poniżej ~0,1–0,2 dB nie są rozstrzygające bez sekcji „Seedy”.</li>
<li>Maska „dno = dolne 40% kadru” jest zgrubna, ale stała dla wszystkich runów.</li>
<li>Ablacja komórek nie douczała Gaussów; mierzy zależność obecnego rozwiązania od lokalnego medium.</li></ul>
</main>
<script>
(() => {
  document.querySelectorAll('figure[data-chart]').forEach(fig => {
    const data = JSON.parse(fig.dataset.chart), svg = fig.querySelector('svg'), hit = svg.querySelector('.hit');
    const cross = svg.querySelector('.crosshair'), tip = fig.querySelector('.tooltip');
    const iters = [...new Set(data.series.flatMap(s => s.curve.map(p => p[0])))].sort((a, b) => a - b);
    const sx = x => data.left + (x - data.x0) / Math.max(data.x1 - data.x0, 1) * (data.width - data.left - data.right);
    function show(event) {
      const box = svg.getBoundingClientRect(), scale = data.width / box.width;
      const px = (event.clientX - box.left) * scale;
      const it = iters.reduce((a, b) => Math.abs(sx(b) - px) < Math.abs(sx(a) - px) ? b : a);
      cross.setAttribute('x1', sx(it)); cross.setAttribute('x2', sx(it)); cross.setAttribute('visibility', 'visible');
      const rows = data.series.map(s => [s, (s.curve.find(p => p[0] === it) || [])[1]]).filter(r => r[1] !== undefined)
        .sort((a, b) => b[1] - a[1]);
      tip.innerHTML = `<b>iteracja ${it}</b><br>` + rows.map(([s, v]) =>
        `<i class="s${s.slot}"></i>${s.name}: ${v.toFixed(data.digits ?? 3).replace('.', ',')} ${data.unit ?? 'dB'}`).join('<br>');
      tip.hidden = false;
      const fbox = fig.getBoundingClientRect(); let left = event.clientX - fbox.left + 14;
      if (left + tip.offsetWidth > fbox.width) left = event.clientX - fbox.left - tip.offsetWidth - 14;
      tip.style.left = Math.max(4, left) + 'px'; tip.style.top = (event.clientY - fbox.top + 10) + 'px';
    }
    hit.addEventListener('pointermove', show);
    hit.addEventListener('pointerleave', () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); });
  });
  document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    document.querySelectorAll('.panel').forEach(p => p.hidden = p.dataset.view !== button.dataset.view);
  }));
})();
</script></body></html>
"""

if __name__ == "__main__":
    main()
