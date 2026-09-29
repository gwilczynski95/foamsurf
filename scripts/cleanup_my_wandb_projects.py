#!/usr/bin/env python3
"""Plan and (only with explicit confirmation) delete my W&B projects.

Scope is deliberately fixed: projects owned by the authenticated W&B user in
mlpolska, excluding foamsurf_iui3_rgb_conflict. The deletion mutation is an
UNDOCUMENTED W&B GraphQL operation, so run `plan` and inspect its JSON first.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ENTITY = "mlpolska"
KEEP = "foamsurf_iui3_rgb_conflict"
PLAN_VERSION = 1
DELETE_MUTATION = """
mutation DeleteOneFoamSurfProject($id: String!) {
  deleteModel(input: {id: $id}) { success }
}
"""


def inventory(api: Any) -> tuple[str, list[dict[str, str]], bool]:
    """Return projects belonging to the signed-in user; never infer from name."""
    username = api.viewer.username
    owned: list[dict[str, str]] = []
    protected_found = False
    for project in api.projects(entity=ENTITY):
        if project.name == KEEP:
            protected_found = True
            continue
        if project.owner.username == username:
            owned.append({"name": project.name, "id": project.id})
    owned.sort(key=lambda item: item["name"])
    if len({item["name"] for item in owned}) != len(owned):
        raise RuntimeError("Duplicate project names in W&B inventory; aborting")
    return username, owned, protected_found


def make_plan(api: Any) -> dict[str, Any]:
    owner, projects, protected_found = inventory(api)
    if not protected_found:
        raise RuntimeError(f"Protected project {ENTITY}/{KEEP} is missing; aborting")
    if not projects:
        raise RuntimeError("No eligible projects; refusing to create an empty plan")
    return {
        "version": PLAN_VERSION,
        "entity": ENTITY,
        "owner": owner,
        "keep": KEEP,
        "projects": projects,
    }


def validate_plan(plan: Any, api: Any) -> list[dict[str, str]]:
    """Fail closed if the account, scope, IDs, or protected project changed."""
    if not isinstance(plan, dict) or (
        plan.get("version") != PLAN_VERSION
        or plan.get("entity") != ENTITY
        or plan.get("keep") != KEEP
        or plan.get("owner") != api.viewer.username
    ):
        raise RuntimeError("Plan version/account/entity/protected project mismatch")
    planned = plan.get("projects")
    if not isinstance(planned, list) or not planned:
        raise RuntimeError("Plan contains no projects")
    if any(
        not isinstance(item, dict)
        or set(item) != {"name", "id"}
        or not isinstance(item["name"], str)
        or not isinstance(item["id"], str)
        or not item["name"]
        or not item["id"]
        or item["name"] == KEEP
        for item in planned
    ):
        raise RuntimeError("Invalid or protected project in plan")
    planned_map = {item["name"]: item["id"] for item in planned}
    if len(planned_map) != len(planned):
        raise RuntimeError("Duplicate names in plan")
    owner, live, protected_found = inventory(api)
    if owner != plan["owner"] or not protected_found:
        raise RuntimeError("Account or protected project changed")
    live_map = {item["name"]: item["id"] for item in live}
    unexpected = sorted(set(live_map) - set(planned_map))
    mismatched = sorted(
        name for name in live_map.keys() & planned_map.keys()
        if live_map[name] != planned_map[name]
    )
    if unexpected or mismatched:
        raise RuntimeError(
            f"Live projects differ from plan: new={unexpected}, changed_ID={mismatched}"
        )
    # Missing projects were already deleted, possibly in an earlier partial run.
    return [item for item in planned if item["name"] in live_map]


def delete_project(api: Any, project_id: str) -> None:
    """W&B internal GraphQL mutation; not part of its supported public SDK."""
    from wandb_gql import gql

    result = api.client.execute(
        gql(DELETE_MUTATION), variable_values={"id": project_id}
    )
    if result.get("deleteModel", {}).get("success") is not True:
        raise RuntimeError(f"W&B did not confirm deletion of project ID {project_id}")


def write_plan(path: Path, plan: dict[str, Any]) -> None:
    # O_EXCL prevents silently replacing a plan that the user already reviewed.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="Read-only W&B inventory and JSON plan")
    plan_parser.add_argument("--out", type=Path, required=True)
    delete_parser = commands.add_parser("delete", help="Delete exactly the reviewed projects")
    delete_parser.add_argument("--plan", type=Path, required=True)
    delete_parser.add_argument(
        "--only", metavar="PROJECT", help="First try one named project from the plan"
    )
    args = parser.parse_args()

    import wandb

    api = wandb.Api(timeout=30)
    if args.command == "plan":
        plan = make_plan(api)
        write_plan(args.out, plan)
        print(f"Saved {len(plan['projects'])} projects to {args.out} (NO DELETIONS)")
        for item in plan["projects"]:
            print(f"  https://wandb.ai/{ENTITY}/{item['name']}")
        print(f"PROTECTED: https://wandb.ai/{ENTITY}/{KEEP}")
        return

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    remaining = validate_plan(plan, api)
    if args.only is not None:
        if args.only not in {item["name"] for item in plan["projects"]}:
            raise SystemExit(f"{args.only!r} is not in the reviewed plan")
        remaining = [item for item in remaining if item["name"] == args.only]
    if not remaining:
        print("Nothing left to delete in this selection.")
        return
    print(f"Will DELETE {len(remaining)} projects owned by {plan['owner']} in {ENTITY}:")
    for item in remaining:
        print(f"  {item['name']}")
    print(f"WILL KEEP: {KEEP}")
    phrase = f"DELETE {len(remaining)} PROJECTS IN {ENTITY}"
    if input(f"Type exactly {phrase!r} to proceed: ").strip() != phrase:
        raise SystemExit("Cancelled; nothing deleted")
    for index, item in enumerate(remaining, 1):
        # Recheck the exact project immediately before the irreversible call.
        current = api.project(item["name"], entity=ENTITY)
        if current.id != item["id"] or current.owner.username != plan["owner"]:
            raise RuntimeError(f"Project changed before deletion: {item['name']}")
        delete_project(api, item["id"])
        print(f"[{index}/{len(remaining)}] DELETED {ENTITY}/{item['name']}", flush=True)


if __name__ == "__main__":
    main()
