#!/usr/bin/env python3
"""Content-addressed verification gate. No unchecked bypass for experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

# K64 warmup and K16 renderer extensions register the same pybind11 class.
# Keep both regression suites, but load them in separate Python processes.
TEST_GROUPS = (
    ("warmup", ("tests/warmup_radfoam_like",
                "tests/b1/test_full_image_sampler.py",
                "tests/b3/test_voronoi_prior_warmup.py")),
    ("renderer", ("renderer/tests/test_native_hybrid_renderer.py",
                  "renderer/tests/test_b01_diagnostics.py")),
)


def fingerprint():
    h = hashlib.sha256()
    folders = ("training", "foamsurf_model", "foamsurf_data", "evaluation", "renderer/native",
               "renderer/foamsurf", "renderer/tests", "tests", "scripts", "configs")
    for folder in folders:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".cu", ".cuh", ".cpp", ".h", ".yaml", ".sbatch", ".sh"}:
                h.update(str(path.relative_to(ROOT)).encode())
                h.update(path.read_bytes())
    return h.hexdigest()


def require_certificate(path):
    data = json.loads(Path(path).read_text())
    if data.get("status") != "passed" or data.get("fingerprint") != fingerprint():
        raise RuntimeError("Tests missing/stale: submit test_warmup_organic.sbatch for this source tree")
    if "4090" not in data.get("gpu", ""):
        raise RuntimeError("Test certificate is not from an RTX4090")
    return data


def run_test_groups(target):
    combined = ET.Element("testsuites")
    commands = []
    returncode = 0
    skipped = 0
    for name, paths in TEST_GROUPS:
        report = target.with_name(f"{target.stem}.{name}.xml")
        # Do not accept an old report if pytest crashes before writing a new one.
        report.unlink(missing_ok=True)
        command = [sys.executable, "-m", "pytest", "-q", *paths,
                   "--junitxml", str(report)]
        commands.append(command)
        print(f"Running isolated test group: {name}", flush=True)
        result = subprocess.run(command, cwd=ROOT)
        returncode = returncode or result.returncode
        try:
            suites = ET.parse(report).getroot()
            found = list(suites.iter("testsuite"))
            if not found or not sum(int(s.get("tests", 0)) for s in found):
                returncode = returncode or 1
            for suite in found:
                skipped += int(suite.get("skipped", 0))
                if int(suite.get("failures", 0)) or int(suite.get("errors", 0)):
                    returncode = returncode or 1
                combined.append(suite)
        except (OSError, ET.ParseError):
            returncode = returncode or 1
    ET.ElementTree(combined).write(target.with_suffix(".xml"), encoding="utf-8",
                                  xml_declaration=True)
    return returncode, skipped, commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["test", "check"])
    parser.add_argument("--certificate", required=True)
    args = parser.parse_args()
    if args.action == "check":
        print(json.dumps(require_certificate(args.certificate), indent=2))
        return
    import torch
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name():
        raise RuntimeError("Tests require the RTX4090 sbatch node")
    before = fingerprint()
    target = Path(args.certificate)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"status": "running", "fingerprint": before}))
    returncode, skipped, commands = run_test_groups(target)
    if returncode or fingerprint() != before:
        target.write_text(json.dumps({"status": "failed", "fingerprint": before,
                                      "returncode": returncode}))
        raise SystemExit(returncode or 1)
    if skipped:
        target.write_text(json.dumps({"status": "failed", "reason": "skipped required tests",
                                      "skipped": skipped}))
        raise SystemExit(1)
    target.write_text(json.dumps({"status": "passed", "fingerprint": before,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "time": time.time(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "commands": commands}, indent=2) + "\n")


if __name__ == "__main__":
    main()
