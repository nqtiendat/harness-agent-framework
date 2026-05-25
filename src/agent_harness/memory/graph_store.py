"""Structured knowledge graph store."""

from __future__ import annotations

import networkx as nx

from agent_harness.memory.semantic import SemanticFact


class KnowledgeGraphStore:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_fact(self, fact: SemanticFact) -> None:
        self.graph.add_edge(fact.subject, fact.object, predicate=fact.predicate, confidence=fact.confidence)

    def facts_for(self, subject: str) -> list[dict]:
        return [
            {"subject": u, "predicate": data.get("predicate"), "object": v, "confidence": data.get("confidence")}
            for u, v, data in self.graph.out_edges(subject, data=True)
        ]

