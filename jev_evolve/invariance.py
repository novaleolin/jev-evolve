"""Does the answer depend on things that should not change it?

A typed decision names its options, so the options can be presented in any
order and the answer should not move. Measured on an 8-option banking task,
it moves a great deal: the same schema, the same model and the same items
scored 0.260 under one option ordering and 0.521 under an average of two.
Reversing a two-option question moved accuracy 14.6 points and flipped the
prediction distribution from 89:7 to 37:59.

That matters beyond the number. A schema whose answer turns on option order
is not measuring its criteria text, so every text mutation is fighting
position noise, and a search over such a schema reports nothing but its own
selection floor. This is the check to run before any tuning, and the fix to
apply before believing any result.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from .backends import Backend
from .policy import Policy
from .trace import Trace


def permutations_of(policy: Policy, k: int, seed: int = 0) -> list:
    """k orderings of every choice point's options, the first as written."""
    rng = random.Random(seed)
    out = [policy]
    for _ in range(k - 1):
        p = policy.copy()
        for pt in p.points.values():
            crit = pt.question.get("criteria")
            if not crit:
                continue
            keys = list(crit)
            rng.shuffle(keys)
            pt.question["criteria"] = {x: crit[x] for x in keys}
        out.append(p)
    return out


@dataclass
class Sensitivity:
    """How much of the answer was the option order, and how much was noise."""

    k: int
    n: int
    flips: int
    control_flips: int = 0
    scores: list = field(default_factory=list)
    control_scores: list = field(default_factory=list)

    @property
    def flip_rate(self) -> float:
        return self.flips / self.n if self.n else 0.0

    @property
    def noise_rate(self) -> float:
        """Answers that move with the ordering held fixed."""
        return self.control_flips / self.n if self.n else 0.0

    @property
    def excess_flip_rate(self) -> float:
        """The part attributable to option order rather than to the backend."""
        return max(0.0, self.flip_rate - self.noise_rate)

    @property
    def spread(self) -> float:
        return (max(self.scores) - min(self.scores)) if self.scores else 0.0

    @property
    def control_spread(self) -> float:
        return ((max(self.control_scores) - min(self.control_scores))
                if self.control_scores else 0.0)

    @property
    def excess_spread(self) -> float:
        return max(0.0, self.spread - self.control_spread)

    @property
    def sound(self) -> bool:
        """Whether a text search on this schema can mean anything.

        The bar is deliberately loose, and it is on the excess. It is not "no
        flips", which no sampling backend reaches; it is "option order moves
        fewer answers than the search is trying to fix".

        The verdict rests on the flip rate alone. Spread is max minus min
        over k runs, which grows with k on its own and is badly resolved at
        the small k anyone will actually pay for, while the flip rate is
        measured over every item. Spread stays in the report as context and
        gets no vote.
        """
        return self.excess_flip_rate < 0.1

    def report(self) -> str:
        L = [f"  option orderings tried   {self.k}"
             f"   (plus {self.k} controls with the order held fixed)",
             f"  answers that changed     {self.flips}/{self.n}"
             f"   ({self.flip_rate:.1%})",
             f"    of which noise alone   {self.control_flips}/{self.n}"
             f"   ({self.noise_rate:.1%})",
             f"    attributable to order  {self.excess_flip_rate:.1%}"]
        if self.scores:
            L += [f"  accuracy by ordering     {min(self.scores):.3f} to"
                  f" {max(self.scores):.3f}   spread {self.spread:.3f}",
                  f"    control spread         {self.control_spread:.3f}"]
        L.append("")
        if self.sound:
            L.append("  STABLE: option order is not deciding the answer")
        else:
            L.append("  ORDER-DEPENDENT: this schema is partly measuring option")
            L.append("  position, so tuning its text will mostly fit noise.")
            L.append("  Wrap the backend in Marginalized(backend, k) and re-check.")
        return "\n".join(L)

    def __str__(self) -> str:
        return self.report()


def permutation_sensitivity(policy: Policy, backend: Backend, tasks: Sequence,
                            score=None, k: int = 4, seed: int = 0,
                            act=None, route=None) -> Sensitivity:
    """Run the same schema under k option orderings, against a control.

    The control matters more than it looks. Counting how often the answer
    changes across k permuted runs also counts every source of
    nondeterminism the backend has, and a sampling model will flip plenty of
    answers with the options left exactly where they were. Reported on its
    own, that number blames option order for noise. So this runs k more times
    with the ordering untouched and reports both, and the verdict is about
    the excess.

    Costs `2 * k * len(tasks)` decisions, and is still worth it before any
    tuning: it is the difference between a search that found nothing and a
    search that could not have found anything.
    """
    from .evolve import run_policy

    sc = score or (lambda e: 0.0)
    perm = [run_policy(p, backend, tasks, sc, act, route)
            for p in permutations_of(policy, k, seed)]
    ctrl = [run_policy(policy, backend, tasks, sc, act, route)
            for _ in range(k)]

    def flips(runs):
        return sum(int(len({tuple(d.choice for d in r.episodes[i].decisions)
                            for r in runs}) > 1) for i in range(len(tasks)))

    return Sensitivity(k=k, n=len(tasks), flips=flips(perm),
                       control_flips=flips(ctrl),
                       scores=[r.score for r in perm] if score else [],
                       control_scores=[r.score for r in ctrl] if score else [])


class Marginalized:
    """Ask each question under several option orderings and average.

    The fix for position bias is old and well understood. What is new is that
    it is affordable. It costs `k` decisions per answer, which is absurd at
    three seconds a call and routine at 450 milliseconds with a tight tail,
    and that is the concrete reason to serve typed decisions from a model
    built for them.

    Averaging the probability each option receives, rather than voting on the
    argmax, keeps the result calibrated enough to threshold on, which the
    abstention machinery downstream depends on.
    """

    def __init__(self, backend: Backend, k: int = 4, seed: int = 0):
        self.backend = backend
        self.k = k
        self._seed = seed
        self.calls = 0

    def decide(self, state: dict, questions: dict) -> dict[str, Any]:
        rng = random.Random(self._seed)
        totals: dict = {}
        nouls: dict = {}
        for _ in range(self.k):
            shuffled = {}
            for name, q in questions.items():
                crit = q.get("criteria")
                if not crit:
                    shuffled[name] = q
                    continue
                keys = list(crit)
                rng.shuffle(keys)
                shuffled[name] = dict(q, criteria={x: crit[x] for x in keys})
            ans = self.backend.decide(state, shuffled)
            self.calls += 1
            for name, a in ans.items():
                if "noul" in a:
                    nouls.setdefault(name, []).append(float(a["noul"]))
                    continue
                acc = totals.setdefault(name, {})
                for opt, p in (a.get("probabilities") or {}).items():
                    acc[opt] = acc.get(opt, 0.0) + float(p)
        out: dict[str, Any] = {}
        for name, acc in totals.items():
            tot = sum(acc.values()) or 1.0
            probs = {o: v / tot for o, v in acc.items()}
            out[name] = {"probabilities": probs,
                         "choice": max(probs, key=probs.get) if probs else None}
        for name, vs in nouls.items():
            out[name] = {"noul": sum(vs) / len(vs)}
        return out


# ------------------------------------------------- irrelevant state


@dataclass
class StateSensitivity:
    """How much the answer moved when irrelevant fields were added."""

    n: int
    k: int
    flips: int
    control_flips: int = 0
    base_score: float = 0.0
    padded_scores: list = field(default_factory=list)

    @property
    def flip_rate(self) -> float:
        return self.flips / self.n if self.n else 0.0

    @property
    def noise_rate(self) -> float:
        return self.control_flips / self.n if self.n else 0.0

    @property
    def excess_flip_rate(self) -> float:
        return max(0.0, self.flip_rate - self.noise_rate)

    @property
    def padded_score(self) -> float:
        """Mean accuracy across the padded runs."""
        if not self.padded_scores:
            return 0.0
        return sum(self.padded_scores) / len(self.padded_scores)

    @property
    def cost(self) -> float:
        """Accuracy lost to the padding. Positive means the padding hurt."""
        return self.base_score - self.padded_score

    @property
    def sound(self) -> bool:
        return self.excess_flip_rate < 0.1

    def report(self) -> str:
        L = [f"  padded variants tried    {self.k}"
             f"   (plus {self.k} unpadded controls)",
             f"  answers that changed     {self.flips}/{self.n}"
             f"   ({self.flip_rate:.1%})",
             f"    of which noise alone   {self.control_flips}/{self.n}"
             f"   ({self.noise_rate:.1%})",
             f"    attributable to padding {self.excess_flip_rate:.1%}",
             f"  accuracy, clean          {self.base_score:.3f}",
             f"  accuracy, padded         {self.padded_score:.3f}"
             f"   ({-self.cost:+.3f})", ""]
        if self.sound:
            L.append("  ROBUST: irrelevant state is not moving the answer")
        else:
            L.append("  STATE-SENSITIVE: fields that cannot bear on this decision")
            L.append("  are changing it. Narrow the point's state_fields, or")
            L.append("  search over them with mutate_state_fields.")
        return "\n".join(L)

    def __str__(self) -> str:
        return self.report()


#: Plausible-looking fields that cannot bear on a decision about the text.
#: Real-looking on purpose: a decision model that ignores obvious filler may
#: still be moved by something that reads like a business field.
DISTRACTORS = {
    "session_id": "a7f3c9e2-4b18-4d55-9c01-6e2f8ab31d94",
    "locale": "en-GB",
    "client_version": "4.11.2",
    "ab_bucket": "control",
    "referrer": "app/home/cards",
    "queue_depth": "37",
    "agent_shift": "evening",
    "last_login_days": "3",
}


def state_sensitivity(policy: Policy, backend: Backend, tasks: Sequence,
                      score=None, k: int = 4, fields: int = 3,
                      distractors: dict | None = None, seed: int = 0,
                      act=None, route=None) -> StateSensitivity:
    """Add irrelevant fields to the state and see whether the answer moves.

    Practitioners report that typed decision models follow complex rules
    poorly out of a long context, which makes what goes into the state a real
    design decision rather than a formatting one. This is the same shape of
    check as `permutation_sensitivity`, against the same kind of control: k
    padded runs and k unpadded ones, with the verdict on the excess.

    The padding is drawn from fields that read like real business metadata,
    because a model that shrugs off obvious filler can still be moved by
    something that looks like it belongs.
    """
    from .evolve import run_policy

    pool = list((distractors or DISTRACTORS).items())
    sc = score or (lambda e: 0.0)
    rng = random.Random(seed)

    def pad(tasks_, extra):
        out = []
        for t in tasks_:
            if isinstance(t, dict):
                tid, st, lab = t.get("id"), t["state"], t.get("label")
            else:
                tid, st, lab = t[0], t[1], (t[2] if len(t) > 2 else None)
            out.append((tid, dict(st, **extra), lab))
        return out

    padded, ctrl = [], []
    for _ in range(k):
        extra = dict(rng.sample(pool, min(fields, len(pool))))
        padded.append(run_policy(policy, backend, pad(tasks, extra), sc,
                                 act, route))
        ctrl.append(run_policy(policy, backend, tasks, sc, act, route))

    def flips(runs):
        return sum(int(len({tuple(d.choice for d in r.episodes[i].decisions)
                            for r in runs}) > 1) for i in range(len(tasks)))

    return StateSensitivity(
        n=len(tasks), k=k, flips=flips(padded), control_flips=flips(ctrl),
        base_score=ctrl[0].score if score else 0.0,
        padded_scores=[r.score for r in padded] if score else [])
