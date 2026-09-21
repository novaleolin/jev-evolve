"""What an agent decided, and what happened next.

A self-evolving agent needs its own history in a shape that can be scored
and replayed. Everything here is plain dataclasses and JSON so a trace
survives the process that made it, and so a schema can be re-scored against
old traces without re-running the agent.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Decision:
    """One typed question, its answer, and what the answer cost.

    `probabilities` is kept alongside `choice` rather than discarded. The
    margin between the top two options is the signal an evolving loop needs
    to tell a confident decision from a coin flip, and throwing it away is
    what forces the next generation to re-run the whole episode to learn
    anything.
    """
    step: int
    question: str
    choice: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    noul: float | None = None
    latency_s: float = 0.0

    @property
    def margin(self) -> float:
        """Top probability minus the runner-up. 0 for a two-way tie."""
        if self.noul is not None:
            return abs(self.noul - 0.5) * 2
        ps = sorted(self.probabilities.values(), reverse=True)
        return (ps[0] - ps[1]) if len(ps) > 1 else (ps[0] if ps else 0.0)


@dataclass
class Episode:
    """One task attempt: the decisions taken and how it came out."""
    task_id: str
    decisions: list[Decision] = field(default_factory=list)
    outcome: float | None = None
    label: Any = None
    started: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_decisions(self) -> int:
        return len(self.decisions)

    @property
    def total_latency(self) -> float:
        return sum(d.latency_s for d in self.decisions)

    @property
    def min_margin(self) -> float:
        """The least confident decision in the episode.

        An episode is usually lost at its weakest link rather than on
        average, so this is the field worth sorting by when deciding which
        episodes a new schema should be tested against first.
        """
        return min((d.margin for d in self.decisions), default=0.0)


class Trace:
    """A run's episodes, appendable and reloadable."""

    def __init__(self, episodes: list[Episode] | None = None):
        self.episodes = episodes or []

    def add(self, ep: Episode) -> None:
        self.episodes.append(ep)

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)

    @property
    def score(self) -> float:
        vals = [e.outcome for e in self.episodes if e.outcome is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for e in self.episodes:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str) -> "Trace":
        eps = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                ds = [Decision(**x) for x in d.pop("decisions", [])]
                eps.append(Episode(decisions=ds, **d))
        return cls(eps)
