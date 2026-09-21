"""The thing that evolves: an agent's decision points, as typed questions.

An agent built this way has no free-text reasoning anywhere in its control
flow. Every branch it takes is a `choice` over a named set of options, a
`noul` over yes/no, or a `score`. That makes the agent's policy a data
structure rather than a prompt, and a data structure is something a loop can
mutate, diff, version and roll back.

What is searchable here is deliberately wider than "the prompt": the
instruction text, the per-option criteria, which fields of the state each
decision is allowed to see, and the thresholds applied to the returned
probabilities. Three of those four are invisible to a prompt optimiser.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Point:
    """One decision point in an agent.

    `question` is the wire-format spec: `{type, instructions, criteria?}`.
    `threshold` is the confidence the top option must reach, relative to the
    field itself, before the agent commits; below it the point abstains and
    the agent can fall back. `state_fields` restricts what this point sees.

    Withholding a field is a real design choice with a real cost, and one
    that is easy to get wrong in the confident direction: a decision that
    sees the whole state is not automatically better, because a typed
    decision model's known failure mode is a large irrelevant state.
    """

    question: dict[str, Any]
    threshold: float = 0.0
    state_fields: list[str] | None = None

    @property
    def type(self) -> str:
        return self.question.get("type", "choice")

    @property
    def options(self) -> list[str]:
        return list((self.question.get("criteria") or {}).keys())

    def visible(self, state: dict) -> dict:
        if self.state_fields is None:
            return state
        return {k: v for k, v in state.items() if k in self.state_fields}


@dataclass
class Policy:
    """A whole agent's decision points, plus the order they run in.

    `order` exists so a generation can reorder decisions, which changes what
    each later point sees. It is a cheap mutation with a large effect: asking
    "is this in scope?" before "which tool?" removes most of the options the
    second question has to discriminate between.
    """

    points: dict[str, Point]
    order: list[str] = field(default_factory=list)
    note: str = ""
    parent: str = ""

    def __post_init__(self):
        if not self.order:
            self.order = list(self.points)

    def copy(self) -> "Policy":
        return Policy(
            {k: Point(copy.deepcopy(p.question), p.threshold,
                      None if p.state_fields is None else list(p.state_fields))
             for k, p in self.points.items()},
            list(self.order), self.note, self.parent)

    def fingerprint(self) -> str:
        blob = json.dumps(
            {k: [self.points[k].question, self.points[k].threshold,
                 self.points[k].state_fields] for k in sorted(self.points)},
            sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {"order": self.order, "note": self.note, "parent": self.parent,
                "points": {k: {"question": p.question, "threshold": p.threshold,
                               "state_fields": p.state_fields}
                           for k, p in self.points.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        pts = {k: Point(v["question"], v.get("threshold", 0.0),
                        v.get("state_fields"))
               for k, v in d["points"].items()}
        return cls(pts, d.get("order") or [], d.get("note", ""),
                   d.get("parent", ""))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Policy":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def choice(instructions: str, criteria: dict[str, str], threshold: float = 0.0,
           state_fields: list[str] | None = None) -> Point:
    """A decision point that picks one named option."""
    return Point({"type": "choice", "instructions": instructions,
                  "criteria": criteria}, threshold, state_fields)


def noul(instructions: str, threshold: float = 0.5,
         state_fields: list[str] | None = None) -> Point:
    """A yes/no decision point. `threshold` is the probability of yes."""
    return Point({"type": "noul", "instructions": instructions},
                 threshold, state_fields)
