"""Environment perception and Agent-Computer Interface transforms."""

from __future__ import annotations

import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_harness.core.types import Observation


ContentType = Literal["html", "log", "text", "json", "unknown"]


class PerceptionConfig(BaseModel):
    max_signals: int = 12
    max_summary_chars: int = 700
    max_state_chars: int = 4000
    max_dom_depth: int = 4
    max_dom_children: int = 60
    raw_spill_threshold: int = 2000


class CompactNode(BaseModel):
    tag: str
    text: str = ""
    attrs: dict[str, str] = Field(default_factory=dict)
    children: list["CompactNode"] = Field(default_factory=list)


class PerceivedState(BaseModel):
    content_type: ContentType
    summary: str
    signals: list[str] = Field(default_factory=list)
    compact_state: dict[str, Any] = Field(default_factory=dict)
    raw_ref: str | None = None
    confidence: float = 1.0


class _SignalHTMLParser(HTMLParser):
    important_tags = {
        "title",
        "h1",
        "h2",
        "h3",
        "main",
        "article",
        "section",
        "nav",
        "form",
        "button",
        "a",
        "input",
        "textarea",
        "select",
        "table",
        "th",
        "td",
        "li",
        "p",
        "code",
        "pre",
    }
    attr_allowlist = {"id", "class", "name", "type", "href", "role", "aria-label", "placeholder", "value"}

    def __init__(self, config: PerceptionConfig) -> None:
        super().__init__(convert_charrefs=True)
        self.config = config
        self.stack: list[CompactNode] = []
        self.roots: list[CompactNode] = []
        self.signal_texts: list[str] = []
        self.node_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self.stack.append(CompactNode(tag=f"ignored:{tag}"))
            return
        if tag not in self.important_tags:
            self.stack.append(CompactNode(tag="ignored"))
            return
        if self.node_count >= self.config.max_dom_children:
            self.stack.append(CompactNode(tag="ignored"))
            return
        filtered_attrs = {
            key: (value or "")[:120]
            for key, value in attrs
            if key in self.attr_allowlist and value is not None and value.strip()
        }
        node = CompactNode(tag=tag, attrs=filtered_attrs)
        self.node_count += 1
        if self.stack and not self.stack[-1].tag.startswith("ignored"):
            if self._depth(self.stack[-1]) < self.config.max_dom_depth:
                self.stack[-1].children.append(node)
        else:
            self.roots.append(node)
        self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        if self.stack:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        text = _normalize_ws(data)
        if not text or not self.stack:
            return
        node = self.stack[-1]
        if node.tag.startswith("ignored"):
            return
        node.text = _trim(f"{node.text} {text}".strip(), 240)
        if node.tag in {"title", "h1", "h2", "h3", "button", "a", "label", "th", "td", "li", "code", "pre"}:
            self.signal_texts.append(_trim(f"{node.tag}: {text}", 300))

    def _depth(self, node: CompactNode) -> int:
        for depth, parent in enumerate(reversed(self.stack), start=1):
            if parent is node:
                return depth
        return 0


class EnvironmentPerceptionEngine:
    """Convert noisy environment payloads into compact decision-state."""

    log_pattern = re.compile(
        r"(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\b[:\]\s-]*(?P<message>.*)",
        re.IGNORECASE,
    )
    timestamp_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b")
    error_words = {"error", "failed", "failure", "exception", "traceback", "denied", "timeout", "invalid"}
    decision_words = {"created", "updated", "deleted", "permission", "budget", "calibration", "completed", "blocked"}

    def __init__(self, raw_directory: str | Path = "storage/workspaces/raw", config: PerceptionConfig | None = None) -> None:
        self.raw_directory = Path(raw_directory)
        self.raw_directory.mkdir(parents=True, exist_ok=True)
        self.config = config or PerceptionConfig()

    def perceive(self, raw: str | dict[str, Any] | list[Any], source: str = "environment", content_type: str | None = None) -> Observation:
        state = self.to_state(raw, content_type=content_type)
        return Observation(
            summary=state.summary,
            signals=state.signals,
            raw_ref=state.raw_ref,
            source=source,
            metadata={
                "content_type": state.content_type,
                "compact_state": state.compact_state,
                "confidence": state.confidence,
            },
        )

    def to_state(self, raw: str | dict[str, Any] | list[Any], content_type: str | None = None) -> PerceivedState:
        raw_text = self._stringify(raw)
        detected = self._detect_type(raw_text, raw, content_type)
        raw_ref = self._spill_raw(raw_text)

        if detected == "html":
            state = self._from_html(raw_text)
        elif detected == "log":
            state = self._from_logs(raw_text)
        elif detected == "json":
            state = self._from_json(raw)
        else:
            state = self._from_text(raw_text)

        state.content_type = detected
        state.raw_ref = raw_ref
        state.compact_state = self._cap_state(state.compact_state)
        return state

    def _detect_type(self, raw_text: str, raw: Any, content_type: str | None) -> ContentType:
        if content_type in {"html", "log", "text", "json"}:
            return content_type  # type: ignore[return-value]
        stripped = raw_text.lstrip()
        if isinstance(raw, (dict, list)) or stripped.startswith("{") or stripped.startswith("["):
            return "json"
        if re.search(r"<(?:html|body|div|main|article|section|button|a|form)\b", raw_text, re.IGNORECASE):
            return "html"
        log_matches = len(self.log_pattern.findall(raw_text))
        if log_matches >= 2 or "Traceback (most recent call last)" in raw_text:
            return "log"
        return "text"

    def _from_html(self, raw_text: str) -> PerceivedState:
        parser = _SignalHTMLParser(self.config)
        parser.feed(raw_text)
        signals = self._rank_signals(parser.signal_texts)
        compact_dom = [root.model_dump(exclude_none=True) for root in parser.roots[: self.config.max_dom_children]]
        title = next((signal.split(":", 1)[1].strip() for signal in signals if signal.startswith("title:")), None)
        summary = title or (signals[0] if signals else "HTML document with no salient interactive signals.")
        return PerceivedState(
            content_type="html",
            summary=_trim(summary, self.config.max_summary_chars),
            signals=signals,
            compact_state={"kind": "dom_tree", "nodes": compact_dom, "node_count": parser.node_count},
            confidence=0.9,
        )

    def _from_logs(self, raw_text: str) -> PerceivedState:
        level_counts: Counter[str] = Counter()
        important: list[dict[str, str]] = []
        for line in raw_text.splitlines():
            clean = _normalize_ws(line)
            if not clean:
                continue
            match = self.log_pattern.search(clean)
            level = (match.group("level").upper() if match else "INFO").replace("WARNING", "WARN")
            message = match.group("message").strip() if match else clean
            level_counts[level] += 1
            score = self._score_text(clean)
            if level in {"ERROR", "CRITICAL", "FATAL", "WARN"} or score >= 2:
                important.append({"level": level, "message": _trim(message, 500)})

        ordered = sorted(important, key=lambda item: self._log_level_weight(item["level"]), reverse=True)
        signals = [f"{item['level']}: {item['message']}" for item in ordered[: self.config.max_signals]]
        summary = signals[0] if signals else "Logs contain no high-priority warnings or errors."
        return PerceivedState(
            content_type="log",
            summary=_trim(summary, self.config.max_summary_chars),
            signals=signals,
            compact_state={"kind": "log_summary", "level_counts": dict(level_counts), "important_events": ordered[:20]},
            confidence=0.85,
        )

    def _from_json(self, raw: Any) -> PerceivedState:
        data = raw
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return self._from_text(raw)
        compact = self._compact_json(data)
        signals = self._json_signals(compact)
        summary = signals[0] if signals else "Structured JSON state parsed."
        return PerceivedState(
            content_type="json",
            summary=_trim(summary, self.config.max_summary_chars),
            signals=signals,
            compact_state={"kind": "json_state", "value": compact},
            confidence=0.95,
        )

    def _from_text(self, raw_text: str) -> PerceivedState:
        candidates = [_normalize_ws(line) for line in raw_text.splitlines() if _normalize_ws(line)]
        if len(candidates) < 3:
            candidates = re.split(r"(?<=[.!?])\s+", raw_text)
        ranked = sorted(
            [candidate for candidate in candidates if candidate.strip()],
            key=self._score_text,
            reverse=True,
        )
        signals = [_trim(item, 500) for item in ranked[: self.config.max_signals]]
        summary = signals[0] if signals else "No salient text signal."
        return PerceivedState(
            content_type="text",
            summary=_trim(summary, self.config.max_summary_chars),
            signals=signals,
            compact_state={"kind": "text_signals", "signal_count": len(signals)},
            confidence=0.75,
        )

    def _rank_signals(self, signals: list[str]) -> list[str]:
        unique = list(dict.fromkeys(_normalize_ws(signal) for signal in signals if _normalize_ws(signal)))
        return sorted(unique, key=self._score_text, reverse=True)[: self.config.max_signals]

    def _score_text(self, text: str) -> int:
        lower = text.lower()
        score = 0
        score += sum(3 for word in self.error_words if word in lower)
        score += sum(2 for word in self.decision_words if word in lower)
        score += 2 if self.timestamp_pattern.search(text) else 0
        score += 1 if len(text) < 240 else 0
        return score

    def _log_level_weight(self, level: str) -> int:
        return {"FATAL": 6, "CRITICAL": 5, "ERROR": 4, "WARN": 3, "INFO": 2, "DEBUG": 1, "TRACE": 0}.get(level, 1)

    def _compact_json(self, value: Any, depth: int = 0) -> Any:
        if depth >= 4:
            return _trim(str(value), 160)
        if isinstance(value, dict):
            result = {}
            for key, item in list(value.items())[:40]:
                result[str(key)] = self._compact_json(item, depth + 1)
            return result
        if isinstance(value, list):
            return [self._compact_json(item, depth + 1) for item in value[:20]]
        if isinstance(value, str):
            return _trim(value, 300)
        return value

    def _json_signals(self, compact: Any) -> list[str]:
        signals: list[str] = []

        def walk(value: Any, path: str = "$") -> None:
            if len(signals) >= self.config.max_signals:
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    key_lower = key.lower()
                    if key_lower in {"error", "status", "state", "decision", "message", "summary", "result"}:
                        signals.append(f"{path}.{key}: {_trim(str(item), 300)}")
                    walk(item, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value[:5]):
                    walk(item, f"{path}[{index}]")

        walk(compact)
        return list(dict.fromkeys(signals))[: self.config.max_signals]

    def _cap_state(self, state: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(state, ensure_ascii=True)
        if len(encoded) <= self.config.max_state_chars:
            return state
        return {
            "kind": state.get("kind", "compact_state"),
            "truncated": True,
            "preview": encoded[: self.config.max_state_chars],
        }

    def _spill_raw(self, raw_text: str) -> str | None:
        if len(raw_text) <= self.config.raw_spill_threshold:
            return None
        path = self.raw_directory / f"observation-{abs(hash(raw_text))}.txt"
        path.write_text(raw_text, encoding="utf-8")
        return str(path)

    def _stringify(self, raw: str | dict[str, Any] | list[Any]) -> str:
        if isinstance(raw, str):
            return raw
        return json.dumps(raw, ensure_ascii=False, indent=2)


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _trim(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."

