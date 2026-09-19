"""CPU regressions for process isolation and fail-closed GPU certification."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


@pytest.mark.parametrize("outcome", ["passed", "failed", "skipped", "missing", "malformed", "empty"])
def test_isolated_gate_reports(tmp_path, monkeypatch, outcome):
    path = Path(__file__).resolve().parents[2] / "scripts/warmup_gate.py"
    spec = importlib.util.spec_from_file_location("warmup_gate_test", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    calls = []

    def run(command, cwd):
        calls.append(command)
        assert cwd == gate.ROOT
        report = Path(command[-1])
        if len(calls) == 1 and outcome == "missing":
            return SimpleNamespace(returncode=1)
        if len(calls) == 1 and outcome == "malformed":
            report.write_text("broken xml")
            return SimpleNamespace(returncode=0)
        first = len(calls) == 1
        skipped = int(first and outcome == "skipped")
        failures = int(first and outcome == "failed")
        tests = 0 if first and outcome == "empty" else 2
        report.write_text(f'<testsuites><testsuite tests="{tests}" skipped="{skipped}" '
                          f'failures="{failures}" errors="0"/></testsuites>')
        return SimpleNamespace(returncode=failures)

    monkeypatch.setattr(gate.subprocess, "run", run)
    target = tmp_path / "passed.json"
    # A stale report must not rescue a subprocess that failed to write a report.
    target.with_name("passed.warmup.xml").write_text(
        '<testsuites><testsuite tests="99"/></testsuites>')
    code, skipped, commands = gate.run_test_groups(target)
    assert len(calls) == 2
    assert commands == calls
    assert "tests/warmup_radfoam_like" in calls[0]
    assert not any(p.startswith("renderer/tests/") for p in calls[0])
    assert "renderer/tests/test_native_hybrid_renderer.py" in calls[1]
    assert "renderer/tests/test_b01_diagnostics.py" in calls[1]
    assert (code == 0) == (outcome in {"passed", "skipped"})
    assert skipped == int(outcome == "skipped")
    ET.parse(target.with_suffix(".xml"))
