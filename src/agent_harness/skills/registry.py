"""Skill registry with metadata-first disclosure and script caching."""

from __future__ import annotations

from pathlib import Path

from agent_harness.core.exceptions import SkillLoadError
from agent_harness.skills.loader import SkillLoader
from agent_harness.skills.package import SkillPackage


class SkillRegistry:
    def __init__(self, loader: SkillLoader | None = None) -> None:
        self.loader = loader or SkillLoader()
        self._skills: dict[str, SkillPackage] = {}
        # cache: (skill_name, command_name) -> (sha256, script_text)
        self._script_cache: dict[tuple[str, str], tuple[str, str]] = {}

    def discover(self, directory: str | Path = "skills") -> None:
        root = Path(directory)
        if not root.exists():
            return
        for child in root.iterdir():
            if child.is_dir() and (child / "skill.yaml").exists():
                package = self.loader.load_metadata(child)
                self._skills[package.name] = package
                self._invalidate_cache(package.name)

    def register(self, package: SkillPackage) -> None:
        self._skills[package.name] = package
        self._invalidate_cache(package.name)

    def list_metadata(self) -> list[dict]:
        return [
            package.metadata_view()
            for package in self._skills.values()
            if not package.deprecated and not package.stale
        ]

    def packages(self) -> list[SkillPackage]:
        return list(self._skills.values())

    def get(self, name: str) -> SkillPackage:
        return self._skills[name]

    def load_script(
        self,
        skill_name: str,
        command_name: str | None = None,
        *,
        expected_sha256: str | None = None,
    ) -> str:
        """Load a script body, validating an optional integrity hash.

        If `expected_sha256` is supplied (typically the hash the caller saw in
        the metadata view), the current on-disk hash must match — otherwise
        `SkillLoadError` is raised. This catches silent script edits between
        discovery and invocation.
        """
        package = self.get(skill_name)
        command = command_name or (package.commands[0].name if package.commands else None)
        if command is None:
            return ""
        cached = self._script_cache.get((skill_name, command))
        current_hash = package.script_hash(command)
        if cached is not None and cached[0] == current_hash:
            script = cached[1]
        else:
            script = self.loader.load_script(package, command)
            if current_hash is not None:
                self._script_cache[(skill_name, command)] = (current_hash, script)
        if expected_sha256 is not None and current_hash != expected_sha256:
            raise SkillLoadError(
                f"Skill integrity mismatch: {skill_name}/{command} "
                f"(expected {expected_sha256}, got {current_hash})"
            )
        return script

    def _invalidate_cache(self, skill_name: str) -> None:
        for key in list(self._script_cache):
            if key[0] == skill_name:
                self._script_cache.pop(key, None)
