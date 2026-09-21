"""The agent loop.

Small on purpose. An agent here is a policy, a backend, and two callbacks:
`route` picks the next decision point, `act` applies the answer to the state.
Everything else, including whatever tools the agent calls, lives in `act` and
is none of this package's business.

The loop's one real job is to make every branch typed and every branch
recorded. An agent that decides by parsing generated text cannot be evolved,
because there is nothing to mutate but a prompt and nothing to score but the
final answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .backends import Backend, timed
from .policy import Point, Policy
from .trace import Decision, Episode

STOP = "__stop__"


@dataclass
class Answer:
    """What one decision point returned, after thresholding."""

    name: str
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    noul: float | None = None
    abstained: bool = False

    @property
    def confidence(self) -> float:
        if self.noul is not None:
            return self.noul
        if not self.probabilities:
            return 0.0
        return max(self.probabilities.values())

    @property
    def margin(self) -> float:
        if self.noul is not None:
            return abs(self.noul - 0.5) * 2
        ps = sorted((self.probabilities or {}).values(), reverse=True)
        return (ps[0] - ps[1]) if len(ps) > 1 else (ps[0] if ps else 0.0)

    def __bool__(self) -> bool:
        """True when the point committed to yes / to an option."""
        if self.abstained:
            return False
        if self.noul is not None:
            return self.noul >= 0.5
        return self.choice is not None


def _apply(point: Point, name: str, raw: dict) -> Answer:
    """Turn a backend answer into a thresholded `Answer`.

    A point below its threshold abstains rather than returning its best
    guess. That distinction is the whole point of having calibrated
    probabilities: an agent that always commits has thrown away the one
    signal that tells it when to ask a human, widen a search, or fall back.
    """
    if point.type == "noul":
        v = float(raw.get("noul", 0.5))
        return Answer(name, noul=v, abstained=abs(v - 0.5) * 2 < 0.0)
    probs = raw.get("probabilities") or {}
    pick = raw.get("choice") or (max(probs, key=probs.get) if probs else None)
    top = max(probs.values()) if probs else 0.0
    if point.threshold and top < point.threshold:
        return Answer(name, choice=None, probabilities=probs, abstained=True)
    return Answer(name, choice=pick, probabilities=probs)


class Agent:
    """Run a policy over a state until it stops.

    `act(state, answer) -> dict | None` applies one answer. Return a dict to
    merge into the state; put `STOP` in it to end the episode early. `route`
    overrides the default behaviour of running `policy.order` once.
    """

    def __init__(self, policy: Policy, backend: Backend,
                 act: Callable[[dict, Answer], Any] | None = None,
                 route: Callable[[dict, list], str | None] | None = None,
                 max_steps: int = 32):
        self.policy = policy
        self.backend = backend
        self.act = act
        self.route = route
        self.max_steps = max_steps

    def _next(self, state: dict, asked: list[str]) -> str | None:
        if self.route is not None:
            return self.route(state, asked)
        for name in self.policy.order:
            if name not in asked:
                return name
        return None

    def run(self, task_id: str, state: dict, label: Any = None) -> Episode:
        state = dict(state)
        ep = Episode(task_id=task_id, label=label)
        asked: list[str] = []
        for step in range(self.max_steps):
            name = self._next(state, asked)
            if name is None:
                break
            point = self.policy.points[name]
            raw, secs = timed(self.backend, point.visible(state),
                              {name: point.question})
            ans = _apply(point, name, raw.get(name, {}))
            asked.append(name)
            ep.decisions.append(Decision(
                step=step, question=name, choice=ans.choice,
                probabilities=ans.probabilities or {}, noul=ans.noul,
                latency_s=secs))
            if self.act is None:
                state[name] = ans.choice if ans.noul is None else ans.noul
                continue
            upd = self.act(state, ans)
            if isinstance(upd, dict):
                stop = upd.pop(STOP, False)
                state.update(upd)
                if stop:
                    break
        ep.meta["final_state"] = {k: v for k, v in state.items()
                                  if isinstance(v, (str, int, float, bool))}
        return ep
