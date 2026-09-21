"""jevolve: agents whose decisions are typed, and whose policy evolves.

Three ideas, in the order you meet them:

1. Every branch the agent takes is a typed question with calibrated
   probabilities, not generated text that something downstream parses.
2. Because the policy is data, a loop can mutate it, and because every
   decision is recorded, the mutations can aim at what actually went wrong.
3. A loop that keeps the best of many candidates reports a gain even when
   nothing improved, so every run states the score a null loop of the same
   size would have reached.

The third is the one that is usually missing.
"""

__version__ = "0.1.0"

from .agent import STOP, Agent, Answer
from .analyze import confusions, cost, overconfident, point_accuracy
from .backends import JevBackend, RuleBackend
from .evolve import Result, evolve, run_policy
from .mutate import default_operators
from .policy import Point, Policy, choice, noul
from .trace import Decision, Episode, Trace

__all__ = [
    "Agent", "Answer", "STOP",
    "Policy", "Point", "choice", "noul",
    "Trace", "Episode", "Decision",
    "RuleBackend", "JevBackend",
    "evolve", "run_policy", "Result", "default_operators",
    "confusions", "point_accuracy", "overconfident", "cost",
]


def __getattr__(name):
    """`LocalBackend` needs torch, so it loads only when asked for.

    Keeping it out of the eager imports is what lets `import jevolve` work in
    an environment with nothing installed but this package and evalfloor.
    """
    if name == "LocalBackend":
        import importlib
        return getattr(importlib.import_module(".backends", __name__), name)
    raise AttributeError(name)
