"""Policy loading helpers."""

from __future__ import annotations

from pathlib import Path

from agent_harness.core.config import load_yaml
from agent_harness.governance.permissions import PermissionController, PermissionDecision


def load_permission_controller(path: str | Path = "configs/permissions.yaml") -> PermissionController:
    data = load_yaml(path)
    controller = PermissionController(PermissionDecision(data.get("default_policy", "ask")))
    for rule in data.get("rules", []):
        controller.add_rule(rule["resource"], rule["decision"])
    return controller

