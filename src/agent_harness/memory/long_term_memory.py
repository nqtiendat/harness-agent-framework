"""Governed long-term memory service."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.core.types import utc_now
from agent_harness.memory.episodic import Episode
from agent_harness.memory.graph_store import KnowledgeGraphStore
from agent_harness.memory.procedural import Procedure
from agent_harness.memory.semantic import SemanticFact
from agent_harness.memory.vector_store import VectorMemoryRecord, VectorMemoryStore


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    DEPRECATED = "deprecated"


class ProjectHistoryRecord(BaseModel):
    project_id: str
    summary: str
    facts: list[SemanticFact] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserProfileRecord(BaseModel):
    user_id: str
    preferences: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    status: MemoryStatus = MemoryStatus.ACTIVE
    updated_at: object = Field(default_factory=utc_now)


class MemoryMaintenanceReport(BaseModel):
    stale_skills: list[str] = Field(default_factory=list)
    stale_memories: list[str] = Field(default_factory=list)
    reason: str = ""


class LongTermMemoryService:
    """Shared context layer for semantic, episodic, and procedural memory."""

    def __init__(self) -> None:
        self.graph = KnowledgeGraphStore()
        self.vectors = VectorMemoryStore()
        self.episodes: list[Episode] = []
        self.procedures: dict[str, Procedure] = {}
        self.user_profiles: dict[str, UserProfileRecord] = {}
        self.project_history: dict[str, ProjectHistoryRecord] = {}
        self.skill_status: dict[str, MemoryStatus] = {}
        self.memory_status: dict[str, MemoryStatus] = {}

    def add_fact(self, fact: SemanticFact, memory_id: str | None = None) -> None:
        self.graph.add_fact(fact)
        if memory_id:
            self.memory_status[memory_id] = MemoryStatus.ACTIVE
        self.vectors.add(
            VectorMemoryRecord(
                id=memory_id or f"fact:{fact.subject}:{fact.predicate}:{fact.object}",
                text=f"{fact.subject} {fact.predicate} {fact.object}",
                metadata={"type": "semantic_fact", "status": MemoryStatus.ACTIVE.value, **fact.model_dump()},
            )
        )

    def add_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)
        memory_id = f"episode:{len(self.episodes)}"
        self.memory_status[memory_id] = MemoryStatus.ACTIVE
        self.vectors.add(VectorMemoryRecord(id=memory_id, text=episode.summary, metadata=episode.model_dump() | {"type": "episode"}))

    def add_procedure(self, procedure: Procedure) -> None:
        self.procedures[procedure.name] = procedure
        self.skill_status[procedure.name] = MemoryStatus.ACTIVE
        self.vectors.add(
            VectorMemoryRecord(
                id=f"procedure:{procedure.name}",
                text=f"{procedure.name} {' '.join(procedure.steps)}",
                metadata={"type": "procedure", "status": MemoryStatus.ACTIVE.value, **procedure.model_dump()},
            )
        )

    def upsert_user_profile(
        self,
        user_id: str,
        preferences: dict[str, Any] | None = None,
        constraints: list[str] | None = None,
    ) -> UserProfileRecord:
        existing = self.user_profiles.get(user_id) or UserProfileRecord(user_id=user_id)
        if preferences:
            existing.preferences.update(preferences)
        if constraints:
            existing.constraints = list(dict.fromkeys([*existing.constraints, *constraints]))
        existing.updated_at = utc_now()
        existing.status = MemoryStatus.ACTIVE
        self.user_profiles[user_id] = existing
        self.memory_status[f"user:{user_id}"] = MemoryStatus.ACTIVE
        self.vectors.add(
            VectorMemoryRecord(
                id=f"user:{user_id}",
                text=f"user profile {user_id} preferences {existing.preferences} constraints {' '.join(existing.constraints)}",
                metadata={"type": "user_profile", **existing.model_dump()},
            )
        )
        return existing

    def upsert_project_history(
        self,
        project_id: str,
        summary: str,
        facts: list[SemanticFact] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProjectHistoryRecord:
        record = self.project_history.get(project_id) or ProjectHistoryRecord(project_id=project_id, summary=summary)
        record.summary = summary
        record.facts = facts or record.facts
        record.tags = list(dict.fromkeys([*record.tags, *(tags or [])]))
        record.metadata.update(metadata or {})
        record.updated_at = utc_now()
        record.status = MemoryStatus.ACTIVE
        self.project_history[project_id] = record
        self.memory_status[f"project:{project_id}"] = MemoryStatus.ACTIVE
        for fact in record.facts:
            self.add_fact(fact, memory_id=f"project:{project_id}:fact:{fact.subject}:{fact.predicate}")
        self.vectors.add(
            VectorMemoryRecord(
                id=f"project:{project_id}",
                text=f"project {project_id} {summary} {' '.join(record.tags)}",
                metadata={"type": "project_history", **record.model_dump()},
            )
        )
        return record

    def retrieve_context(self, query: str, limit: int = 5, include_stale: bool = False) -> list[VectorMemoryRecord]:
        records = self.vectors.search(query, limit=limit * 3)
        filtered: list[VectorMemoryRecord] = []
        for record in records:
            status = record.metadata.get("status", MemoryStatus.ACTIVE.value)
            if include_stale or status == MemoryStatus.ACTIVE.value:
                filtered.append(record)
            if len(filtered) >= limit:
                break
        return filtered

    def mark_skill_stale(self, skill_name: str, reason: str = "") -> None:
        self.skill_status[skill_name] = MemoryStatus.STALE
        procedure = self.procedures.get(skill_name)
        if procedure:
            procedure.steps.append(f"STALE: {reason}" if reason else "STALE")

    def mark_memory_stale(self, memory_id: str, reason: str = "") -> None:
        self.memory_status[memory_id] = MemoryStatus.STALE
        for record in self.vectors._records:
            if record.id == memory_id:
                record.metadata["status"] = MemoryStatus.STALE.value
                record.metadata["stale_reason"] = reason

    def maintenance_for_environment(
        self,
        *,
        current_tool_versions: dict[str, str],
        skill_tool_requirements: dict[str, dict[str, str]],
    ) -> MemoryMaintenanceReport:
        report = MemoryMaintenanceReport(reason="tool_api_version_check")
        for skill_name, requirements in skill_tool_requirements.items():
            stale_reasons = []
            for tool_name, required_version in requirements.items():
                current_version = current_tool_versions.get(tool_name)
                if current_version is None or current_version != required_version:
                    stale_reasons.append(f"{tool_name}: required={required_version}, current={current_version}")
            if stale_reasons:
                self.mark_skill_stale(skill_name, "; ".join(stale_reasons))
                report.stale_skills.append(skill_name)
        return report

