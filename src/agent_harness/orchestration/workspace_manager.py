"""Shared workspace and artifact-mediated collaboration."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_harness.core.exceptions import PermissionDeniedError
from agent_harness.core.types import utc_now
from agent_harness.evaluation.tracer import ErrorSource, ExecutionTracer


ArtifactType = Literal["plan", "code", "test", "report", "data", "conflict", "generic"]


class ArtifactRecord(BaseModel):
    path: str
    artifact_type: ArtifactType = "generic"
    author: str
    revision: int
    content_hash: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceConflict(BaseModel):
    path: str
    base_revision: int | None
    current_revision: int
    attempted_author: str
    conflict_path: str
    reason: str
    created_at: object = Field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class WorkspaceLock:
    path: str
    owner: str
    token: str


class WorkspaceManager:
    """File-backed shared state with read-write asymmetry controls."""

    def __init__(self, root: str | Path = "storage/workspaces/shared", tracer: ExecutionTracer | None = None) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._mutex = RLock()
        self._locks: dict[str, WorkspaceLock] = {}
        self._revisions: dict[str, int] = {}
        self._records: dict[str, ArtifactRecord] = {}
        self._conflicts: list[WorkspaceConflict] = []
        self.tracer = tracer

    def create_artifact(
        self,
        path: str,
        content: str,
        *,
        author: str,
        artifact_type: ArtifactType = "generic",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        return self.write_artifact(
            path,
            content,
            author=author,
            artifact_type=artifact_type,
            expected_revision=0,
            metadata=metadata,
        )

    def read_artifact(self, path: str) -> str:
        artifact_path = self._resolve(path)
        if not artifact_path.exists():
            raise FileNotFoundError(path)
        return artifact_path.read_text(encoding="utf-8")

    def write_artifact(
        self,
        path: str,
        content: str,
        *,
        author: str,
        artifact_type: ArtifactType = "generic",
        expected_revision: int | None = None,
        lock_token: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        with self._mutex:
            normalized = self._normalize(path)
            self._assert_lock_allows_write(normalized, author, lock_token)
            current_revision = self._revisions.get(normalized, 0)
            if expected_revision is not None and expected_revision != current_revision:
                conflict = self._write_conflict(
                    normalized,
                    attempted_content=content,
                    author=author,
                    base_revision=expected_revision,
                    current_revision=current_revision,
                )
                raise WorkspaceConflictError(conflict)

            artifact_path = self._resolve(normalized)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(content, encoding="utf-8")
            revision = current_revision + 1
            self._revisions[normalized] = revision
            # Persist a per-revision snapshot under .versions/<path>/rev.txt so
            # post-mortem can diff any two revisions; conflict artifacts and the
            # versions tree itself are excluded to avoid recursion / noise.
            if not normalized.startswith(".conflicts/") and not normalized.startswith(".versions/"):
                versions_path = self._resolve(f".versions/{normalized}/{revision:06d}.txt")
                versions_path.parent.mkdir(parents=True, exist_ok=True)
                versions_path.write_text(content, encoding="utf-8")
            record = ArtifactRecord(
                path=normalized,
                artifact_type=artifact_type,
                author=author,
                revision=revision,
                content_hash=str(hash(content)),
                metadata=metadata or {},
            )
            self._records[normalized] = record
            if self.tracer:
                self.tracer.record_artifact(record.path, record.author, record.revision, record.artifact_type)
            return record

    def append_artifact(
        self,
        path: str,
        content: str,
        *,
        author: str,
        artifact_type: ArtifactType = "generic",
        expected_revision: int | None = None,
        lock_token: str | None = None,
    ) -> ArtifactRecord:
        existing = self.read_artifact(path) if self.exists(path) else ""
        separator = "" if not existing or existing.endswith("\n") else "\n"
        return self.write_artifact(
            path,
            f"{existing}{separator}{content}",
            author=author,
            artifact_type=artifact_type,
            expected_revision=expected_revision,
            lock_token=lock_token,
        )

    def acquire_lock(self, path: str, owner: str) -> WorkspaceLock:
        with self._mutex:
            normalized = self._normalize(path)
            existing = self._locks.get(normalized)
            if existing and existing.owner != owner:
                raise WorkspaceLockError(f"Artifact is locked by {existing.owner}: {normalized}")
            lock = WorkspaceLock(path=normalized, owner=owner, token=str(uuid4()))
            self._locks[normalized] = lock
            return lock

    def release_lock(self, lock: WorkspaceLock) -> None:
        with self._mutex:
            existing = self._locks.get(lock.path)
            if existing and existing.token == lock.token:
                self._locks.pop(lock.path, None)

    def revision(self, path: str) -> int:
        return self._revisions.get(self._normalize(path), 0)

    def revision_history(self, path: str) -> list[int]:
        """List available revisions persisted under .versions/<path>/."""
        normalized = self._normalize(path)
        versions_dir = self._resolve(f".versions/{normalized}")
        if not versions_dir.exists():
            return []
        return sorted(int(p.stem) for p in versions_dir.glob("*.txt") if p.stem.isdigit())

    def read_revision(self, path: str, revision: int) -> str:
        normalized = self._normalize(path)
        version_path = self._resolve(f".versions/{normalized}/{revision:06d}.txt")
        if not version_path.exists():
            raise FileNotFoundError(f"{path}@rev{revision}")
        return version_path.read_text(encoding="utf-8")

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def list_artifacts(self) -> list[ArtifactRecord]:
        return sorted(self._records.values(), key=lambda record: record.path)

    def list_conflicts(self) -> list[WorkspaceConflict]:
        return list(self._conflicts)

    def resolve_conflict(
        self,
        conflict: WorkspaceConflict,
        resolved_content: str,
        *,
        resolver: str,
    ) -> ArtifactRecord:
        return self.write_artifact(
            conflict.path,
            resolved_content,
            author=resolver,
            artifact_type="generic",
            expected_revision=conflict.current_revision,
        )

    def snapshot_for_agent(self, paths: list[str] | None = None) -> dict[str, Any]:
        records = self.list_artifacts()
        if paths is not None:
            wanted = {self._normalize(path) for path in paths}
            records = [record for record in records if record.path in wanted]
        return {
            "root": str(self.root),
            "artifacts": [record.model_dump() for record in records],
            "conflicts": [conflict.model_dump() for conflict in self._conflicts],
        }

    def _write_conflict(
        self,
        path: str,
        *,
        attempted_content: str,
        author: str,
        base_revision: int | None,
        current_revision: int,
    ) -> WorkspaceConflict:
        current_content = self.read_artifact(path) if self.exists(path) else ""
        conflict_body = self._merge_conflict_text(
            path=path,
            current_content=current_content,
            attempted_content=attempted_content,
            current_revision=current_revision,
            attempted_author=author,
        )
        conflict_path = f".conflicts/{path.replace('/', '__')}.{uuid4().hex}.diff"
        resolved_conflict_path = self._resolve(conflict_path)
        resolved_conflict_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_conflict_path.write_text(conflict_body, encoding="utf-8")
        conflict = WorkspaceConflict(
            path=path,
            base_revision=base_revision,
            current_revision=current_revision,
            attempted_author=author,
            conflict_path=conflict_path,
            reason="expected revision does not match current revision",
        )
        self._conflicts.append(conflict)
        if self.tracer:
            self.tracer.record("artifact.conflict", conflict.model_dump())
            self.tracer.record_error(
                source=ErrorSource.WORKSPACE,
                message=f"workspace conflict for {path}",
                exception_type="WorkspaceConflictError",
                context={"path": path, "attempted_author": author, "current_revision": current_revision},
            )
        self._records[conflict_path] = ArtifactRecord(
            path=conflict_path,
            artifact_type="conflict",
            author="workspace-manager",
            revision=1,
            content_hash=str(hash(conflict_body)),
            metadata=conflict.model_dump(),
        )
        return conflict

    def _merge_conflict_text(
        self,
        *,
        path: str,
        current_content: str,
        attempted_content: str,
        current_revision: int,
        attempted_author: str,
    ) -> str:
        diff = "\n".join(
            difflib.unified_diff(
                current_content.splitlines(),
                attempted_content.splitlines(),
                fromfile=f"{path}@rev{current_revision}",
                tofile=f"{path}@{attempted_author}",
                lineterm="",
            )
        )
        return (
            f"<<<<<<< CURRENT rev {current_revision}\n"
            f"{current_content}\n"
            "=======\n"
            f"{attempted_content}\n"
            f">>>>>>> ATTEMPTED by {attempted_author}\n\n"
            "Unified diff:\n"
            f"{diff}\n"
        )

    def _assert_lock_allows_write(self, path: str, author: str, lock_token: str | None) -> None:
        existing = self._locks.get(path)
        if not existing:
            return
        if existing.owner == author and existing.token == lock_token:
            return
        raise WorkspaceLockError(f"Artifact is locked by {existing.owner}: {path}")

    def _resolve(self, path: str) -> Path:
        resolved = (self.root / self._normalize(path)).resolve()
        if not resolved.is_relative_to(self.root):
            raise PermissionDeniedError(f"Path escapes shared workspace: {resolved}")
        return resolved

    def _normalize(self, path: str) -> str:
        normalized = Path(path.replace("\\", "/")).as_posix().lstrip("/")
        if normalized.startswith("../") or normalized == "..":
            raise PermissionDeniedError(f"Path escapes shared workspace: {path}")
        return normalized


class WorkspaceConflictError(Exception):
    def __init__(self, conflict: WorkspaceConflict) -> None:
        super().__init__(f"Workspace conflict for {conflict.path}; conflict artifact: {conflict.conflict_path}")
        self.conflict = conflict


class WorkspaceLockError(Exception):
    pass
