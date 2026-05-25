"""Progressive skill loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_harness.core.exceptions import SkillLoadError
from agent_harness.skills.package import SkillCommand, SkillPackage


class SkillLoader:
    def load_metadata(self, path: str | Path) -> SkillPackage:
        root = Path(path)
        metadata_file = root / "skill.yaml"
        if not metadata_file.exists():
            raise SkillLoadError(f"Missing skill.yaml: {root}")
        data = yaml.safe_load(metadata_file.read_text(encoding="utf-8")) or {}
        commands = [SkillCommand(**item) for item in data.get("commands", [])]
        return SkillPackage(
            name=data["name"],
            version=str(data["version"]),
            description=data.get("description", ""),
            expected_inputs=data.get("expected_inputs", {}),
            tags=data.get("tags", []),
            tool_requirements=data.get("tool_requirements", {}),
            commands=commands,
            path=root,
            deprecated=bool(data.get("deprecated", False)),
            stale=bool(data.get("stale", False)),
        )

    def load_script(self, package: SkillPackage, command_name: str) -> str:
        command = next((item for item in package.commands if item.name == command_name), None)
        if command is None or not command.script:
            raise SkillLoadError(f"Command has no script: {command_name}")
        if package.path is None:
            raise SkillLoadError(f"Package path is missing: {package.name}")
        script_path = package.path / command.script
        return script_path.read_text(encoding="utf-8")
