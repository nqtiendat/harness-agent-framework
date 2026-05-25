"""Skill package metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field


class SkillCommand(BaseModel):
    name: str
    description: str
    script: str | None = None
    permissions: list[str] = Field(default_factory=list)
    expected_inputs: dict[str, str] = Field(default_factory=dict)


class SkillPackage(BaseModel):
    name: str
    version: str
    description: str
    expected_inputs: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    tool_requirements: dict[str, str] = Field(default_factory=dict)
    commands: list[SkillCommand] = Field(default_factory=list)
    path: Path | None = None
    deprecated: bool = False
    stale: bool = False

    def metadata_view(self) -> dict:
        """Progressive-disclosure view: name/desc/permissions/hash, no script body."""
        return self.model_dump(exclude={"path", "commands"}) | {
            "commands": [
                {
                    "name": command.name,
                    "description": command.description,
                    "permissions": list(command.permissions),
                    "script_sha256": self.script_hash(command.name),
                }
                for command in self.commands
            ],
        }

    def script_hash(self, command_name: str) -> str | None:
        """SHA-256 of the script body for `command_name`, or None if no script.

        Used to detect silent script edits between discovery and invocation —
        callers compare the hash they saw in metadata with `script_hash` after
        re-reading the body to ensure integrity.
        """
        command = next((c for c in self.commands if c.name == command_name), None)
        if command is None or not command.script or self.path is None:
            return None
        script_path = self.path / command.script
        if not script_path.exists():
            return None
        return hashlib.sha256(script_path.read_bytes()).hexdigest()

    def permissions_for(self, command_name: str) -> list[str]:
        command = next((c for c in self.commands if c.name == command_name), None)
        return list(command.permissions) if command else []
