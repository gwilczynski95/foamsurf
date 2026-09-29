"""CPU-only guard tests for scripts/cleanup_my_wandb_projects.py."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/cleanup_my_wandb_projects.py"
SPEC = spec_from_file_location("cleanup_my_wandb_projects", SOURCE)
assert SPEC and SPEC.loader
cleanup = module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


class FakeApi:
    def __init__(self, rows):
        self.viewer = SimpleNamespace(username="me")
        self.rows = rows

    def projects(self, entity):
        assert entity == "mlpolska"
        return [
            SimpleNamespace(name=name, id=project_id,
                            owner=SimpleNamespace(username=owner))
            for name, project_id, owner in self.rows
        ]


BASE = [
    ("foamsurf_iui3_rgb_conflict", "keep-id", "me"),
    ("my-old-project", "old-id", "me"),
    ("coworker-project", "other-id", "coworker"),
]


def test_plan_excludes_protected_and_coworker():
    plan = cleanup.make_plan(FakeApi(BASE))
    assert plan["projects"] == [{"name": "my-old-project", "id": "old-id"}]
    assert plan["keep"] == "foamsurf_iui3_rgb_conflict"


def test_plan_requires_protected_project():
    with pytest.raises(RuntimeError, match="Protected"):
        cleanup.make_plan(FakeApi(BASE[1:]))


def test_validate_rejects_added_project():
    plan = cleanup.make_plan(FakeApi(BASE))
    changed = FakeApi(BASE + [("new-project", "new-id", "me")])
    with pytest.raises(RuntimeError, match="Live projects differ"):
        cleanup.validate_plan(plan, changed)


def test_validate_rejects_id_change():
    plan = cleanup.make_plan(FakeApi(BASE))
    changed = FakeApi([BASE[0], ("my-old-project", "replacement-id", "me"), BASE[2]])
    with pytest.raises(RuntimeError, match="changed_ID"):
        cleanup.validate_plan(plan, changed)


def test_validate_can_resume_after_partial_deletion():
    plan = cleanup.make_plan(FakeApi(BASE + [("older", "older-id", "me")]))
    assert cleanup.validate_plan(plan, FakeApi(BASE)) == [
        {"name": "my-old-project", "id": "old-id"}
    ]


def test_validate_rejects_protected_name_in_plan():
    plan = cleanup.make_plan(FakeApi(BASE))
    plan["projects"].append({"name": cleanup.KEEP, "id": "keep-id"})
    with pytest.raises(RuntimeError, match="Invalid or protected"):
        cleanup.validate_plan(plan, FakeApi(BASE))


def test_write_plan_will_not_overwrite(tmp_path):
    path = tmp_path / "plan.json"
    cleanup.write_plan(path, cleanup.make_plan(FakeApi(BASE)))
    with pytest.raises(FileExistsError):
        cleanup.write_plan(path, cleanup.make_plan(FakeApi(BASE)))
