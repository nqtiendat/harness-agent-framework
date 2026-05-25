"""Skill maintenance helpers."""

from __future__ import annotations

from agent_harness.skills.package import SkillPackage


class SkillMaintenance:
    def stale(self, packages: list[SkillPackage]) -> list[SkillPackage]:
        return [package for package in packages if package.stale]

    def deprecated(self, packages: list[SkillPackage]) -> list[SkillPackage]:
        return [package for package in packages if package.deprecated]

    def retired(self, packages: list[SkillPackage]) -> list[SkillPackage]:
        return [package for package in packages if package.stale or package.deprecated]
