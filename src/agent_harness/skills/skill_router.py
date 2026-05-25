"""Skill retrieval with metadata-only progressive disclosure."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from agent_harness.core.exceptions import PermissionDeniedError
from agent_harness.memory.vector_store import VectorMemoryRecord, VectorMemoryStore
from agent_harness.skills.package import SkillPackage
from agent_harness.skills.registry import SkillRegistry


@dataclass(frozen=True, slots=True)
class SkillRoute:
    name: str
    score: int
    metadata: dict


class SkillRouter:
    """Retrieve relevant skills without loading their full scripts."""

    def __init__(self, registry: SkillRegistry, vector_store: VectorMemoryStore | None = None) -> None:
        self.registry = registry
        self.vector_store = vector_store or VectorMemoryStore()
        self._indexed = False

    def index_registry(self, force: bool = False) -> None:
        if self._indexed and not force:
            return
        self.vector_store = VectorMemoryStore()
        for package in self.registry.packages():
            if package.deprecated or package.stale:
                continue
            self.vector_store.add(self._record_for(package))
        self._indexed = True

    def route(self, task: str, limit: int = 5) -> list[SkillRoute]:
        self.index_registry()
        routes: list[SkillRoute] = []
        for record in self.vector_store.search(task, limit=limit):
            skill_name = record.metadata["skill_name"]
            package = self.registry.get(skill_name)
            routes.append(
                SkillRoute(
                    name=skill_name,
                    score=self._score(task, record.text),
                    metadata=package.metadata_view(),
                )
            )
        return sorted(routes, key=lambda route: route.score, reverse=True)[:limit]

    def load_for_invocation(
        self,
        skill_name: str,
        command_name: str | None = None,
        *,
        allowed_permissions: Iterable[str] | None = None,
    ) -> dict:
        """Load a skill script, enforcing declared permissions and integrity.

        `allowed_permissions` is the set of permissions the calling harness has
        granted for this invocation. If the skill command declares any
        permission outside that set, `PermissionDeniedError` is raised.

        The script body is then loaded with an integrity check against the hash
        recorded in metadata.
        """
        package = self.registry.get(skill_name)
        command = command_name or (package.commands[0].name if package.commands else None)
        if command is None:
            return {"metadata": package.metadata_view(), "command": None, "script": ""}

        required = package.permissions_for(command)
        if allowed_permissions is not None:
            allowed = set(allowed_permissions)
            missing = [perm for perm in required if perm not in allowed]
            if missing:
                raise PermissionDeniedError(
                    f"Skill {skill_name}/{command} requires permissions not granted: {missing}"
                )

        expected_hash = package.script_hash(command)
        script = self.registry.load_script(
            skill_name, command, expected_sha256=expected_hash
        )
        return {
            "metadata": package.metadata_view(),
            "command": command,
            "script": script,
            "permissions": required,
            "script_sha256": expected_hash,
        }

    def _record_for(self, package: SkillPackage) -> VectorMemoryRecord:
        command_text = " ".join(f"{command.name} {command.description}" for command in package.commands)
        expected_inputs = " ".join(f"{key} {value}" for key, value in package.expected_inputs.items())
        text = f"{package.name} {package.description} {' '.join(package.tags)} {expected_inputs} {command_text}"
        return VectorMemoryRecord(
            id=f"skill:{package.name}",
            text=text,
            metadata={"type": "skill_metadata", "skill_name": package.name, "status": "active"},
        )

    def _score(self, task: str, text: str) -> int:
        task_terms = set(task.lower().split())
        text_terms = set(text.lower().split())
        return len(task_terms.intersection(text_terms))
