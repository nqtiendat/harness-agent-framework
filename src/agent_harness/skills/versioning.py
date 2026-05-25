"""Skill version helpers."""

from __future__ import annotations

from packaging.version import Version

from agent_harness.skills.package import SkillPackage


class SkillVersionManager:
    def latest(self, packages: list[SkillPackage]) -> SkillPackage:
        return sorted(packages, key=lambda package: Version(package.version), reverse=True)[0]

