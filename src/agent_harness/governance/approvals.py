"""Approval abstractions for actions requiring human confirmation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_harness.governance.permissions import PermissionRequest


class ApprovalProvider(ABC):
    @abstractmethod
    async def approve(self, request: PermissionRequest) -> bool:
        """Return whether the requested action is approved."""


class DenyByDefaultApprovalProvider(ApprovalProvider):
    async def approve(self, request: PermissionRequest) -> bool:
        return False

