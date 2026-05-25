"""Skill acquisition, packaging, and maintenance."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from agent_harness.core.types import AgentState
from agent_harness.memory.episodic import Episode
from agent_harness.memory.long_term_memory import LongTermMemoryService, MemoryStatus
from agent_harness.memory.procedural import Procedure
from agent_harness.skills.package import SkillCommand, SkillPackage
from agent_harness.skills.registry import SkillRegistry


class SkillAcquisitionInput(BaseModel):
    name: str
    description: str
    decisions: list[str] = Field(default_factory=list)
    calibration_points: list[str] = Field(default_factory=list)
    expected_inputs: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    tool_requirements: dict[str, str] = Field(default_factory=dict)
    source_session_id: str | None = None


class SkillAcquisitionResult(BaseModel):
    package: SkillPackage
    package_path: Path
    script_path: Path
    test_path: Path


class SkillManager:
    """Convert successful episodes into reusable procedural skill packages."""

    def __init__(
        self,
        registry: SkillRegistry,
        long_term_memory: LongTermMemoryService,
        skills_directory: str | Path = "skills",
    ) -> None:
        self.registry = registry
        self.long_term_memory = long_term_memory
        self.skills_directory = Path(skills_directory)
        self.skills_directory.mkdir(parents=True, exist_ok=True)

    def acquire_from_state(
        self,
        state: AgentState,
        *,
        name: str | None = None,
        description: str | None = None,
        expected_inputs: dict[str, str] | None = None,
        tags: list[str] | None = None,
        tool_requirements: dict[str, str] | None = None,
    ) -> SkillAcquisitionResult:
        decisions = list(getattr(self.long_term_memory, "_last_decisions", []))
        if not decisions and state.messages:
            decisions = [message.content for message in state.messages if "completed" in message.content.lower()]
        calibration_points = [
            f"{observation.source}: {observation.summary}"
            for observation in state.observations
            if observation.summary
        ]
        return self.acquire(
            SkillAcquisitionInput(
                name=name or self._name_from_goal(state.goal),
                description=description or f"Reusable procedure learned from goal: {state.goal}",
                decisions=decisions,
                calibration_points=calibration_points,
                expected_inputs=expected_inputs or {"goal": "The task goal or target artifact."},
                tags=tags or ["acquired", "procedural"],
                tool_requirements=tool_requirements or {},
                source_session_id=state.session_id,
            )
        )

    def acquire(self, request: SkillAcquisitionInput) -> SkillAcquisitionResult:
        slug = self._slug(request.name)
        package_path = self.skills_directory / slug
        scripts_path = package_path / "scripts"
        tests_path = package_path / "tests"
        scripts_path.mkdir(parents=True, exist_ok=True)
        tests_path.mkdir(parents=True, exist_ok=True)

        script = self._render_script(request)
        script_path = scripts_path / "run.md"
        script_path.write_text(script, encoding="utf-8")

        test_cases = self._render_tests(request)
        test_path = tests_path / "test_cases.yaml"
        test_path.write_text(yaml.safe_dump(test_cases, sort_keys=False), encoding="utf-8")

        command = SkillCommand(
            name="run",
            description=f"Execute {request.name}.",
            script="scripts/run.md",
            expected_inputs=request.expected_inputs,
        )
        package = SkillPackage(
            name=slug,
            version="0.1.0",
            description=request.description,
            expected_inputs=request.expected_inputs,
            tags=request.tags,
            tool_requirements=request.tool_requirements,
            commands=[command],
            path=package_path,
        )
        metadata = package.model_dump(exclude={"path"})
        metadata["commands"] = [command.model_dump()]
        (package_path / "skill.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
        (package_path / "commands.yaml").write_text(
            yaml.safe_dump({"commands": {"run": {"input": request.expected_inputs, "output": {"result": "string"}}}}, sort_keys=False),
            encoding="utf-8",
        )

        self.registry.register(package)
        self.long_term_memory.add_episode(
            Episode(
                title=f"Skill acquired: {slug}",
                summary=f"{request.description} Decisions: {'; '.join(request.decisions[:5])}",
                session_id=request.source_session_id or "unknown",
            )
        )
        self.long_term_memory.add_procedure(
            Procedure(
                name=slug,
                steps=[*request.decisions, *request.calibration_points],
                source_skill=slug,
            )
        )
        return SkillAcquisitionResult(package=package, package_path=package_path, script_path=script_path, test_path=test_path)

    def mark_stale_by_tool_versions(self, current_tool_versions: dict[str, str]) -> list[str]:
        stale: list[str] = []
        requirements = {package.name: package.tool_requirements for package in self.registry.packages()}
        report = self.long_term_memory.maintenance_for_environment(
            current_tool_versions=current_tool_versions,
            skill_tool_requirements=requirements,
        )
        for skill_name in report.stale_skills:
            package = self.registry.get(skill_name)
            package.stale = True
            stale.append(skill_name)
        return stale

    def prune_stale_metadata(self) -> None:
        for package in self.registry.packages():
            status = self.long_term_memory.skill_status.get(package.name)
            if status is MemoryStatus.STALE:
                package.stale = True

    def _render_script(self, request: SkillAcquisitionInput) -> str:
        decisions = "\n".join(f"{index}. {item}" for index, item in enumerate(request.decisions, start=1)) or "1. Inspect current task constraints."
        calibrations = "\n".join(f"- {item}" for item in request.calibration_points) or "- Verify the result against the environment."
        inputs = "\n".join(f"- `{key}`: {value}" for key, value in request.expected_inputs.items()) or "- `goal`: task goal"
        return (
            f"# {request.name}\n\n"
            f"{request.description}\n\n"
            "## Expected Inputs\n\n"
            f"{inputs}\n\n"
            "## Procedure\n\n"
            f"{decisions}\n\n"
            "## Calibration Points\n\n"
            f"{calibrations}\n\n"
            "## Completion Criteria\n\n"
            "- All calibration points pass.\n"
            "- Important decisions are recorded in working memory.\n"
            "- Rehydratable tool outputs are stored as artifact references, not copied into context.\n"
        )

    def _render_tests(self, request: SkillAcquisitionInput) -> dict[str, Any]:
        return {
            "skill": self._slug(request.name),
            "cases": [
                {
                    "name": "metadata-loads",
                    "input": {key: f"example {key}" for key in request.expected_inputs},
                    "assertions": [
                        "skill metadata is discoverable",
                        "script is loaded only during invocation",
                        "calibration points are present",
                    ],
                }
            ],
        }

    def _name_from_goal(self, goal: str) -> str:
        return "skill-" + self._slug(goal)[:48]

    def _slug(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
        return slug or "acquired-skill"
