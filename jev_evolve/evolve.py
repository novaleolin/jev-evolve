"""The evolution loop, and the part that tells you how much of it was noise.

A generational loop over an eval set is a selection procedure, and the
maximum of many noisy measurements is biased upward. Thirty generations of a
policy that never actually improves still reports a gain, and the gain grows
with the number of generations. That is not a flaw in this loop in
particular; it is arithmetic, and it applies to every self-improving agent
loop that keeps the best-scoring candidate.

So this module does two things. It evolves the policy, and it reports the
score a loop of the same shape would have reached on a policy with no real
improvement in it at all. If the run's gain is under that number, the run
found nothing, and `Result.credible` says so.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .agent import Agent
from .backends import Backend
from .policy import Policy
from .trace import Episode, Trace

#: Scores one finished episode. 0/1 keeps the floor arithmetic exact; a
#: graded score still works, but the reported floor is then a lower bound.
ScoreFn = Callable[[Episode], float]


@dataclass
class Generation:
    index: int
    policy: Policy
    score: float
    note: str
    accepted: bool
    trace: Trace = field(default=None, repr=False)


@dataclass
class Result:
    """What a run found, and whether it found anything."""

    generations: list[Generation]
    base: Policy
    best: Policy
    base_score: float
    best_score: float
    floor: Any = None
    heldout: Any = None
    base_heldout: float | None = None
    best_heldout: float | None = None

    @property
    def gain(self) -> float:
        return self.best_score - self.base_score

    @property
    def credible(self) -> bool:
        """True only on evidence from data the search did not run on.

        The selection floor governs the split that was searched, and nothing
        else. A winner can sit under the floor on the training split and
        still be real, which is exactly what a held-out paired test is for,
        so the floor is reported but does not get a vote here.
        """
        return bool(self.heldout is not None and self.heldout.confirmed)

    @property
    def verdict(self) -> str:
        if self.heldout is None:
            return "UNTESTED"
        if self.heldout.confirmed:
            return "CREDIBLE"
        if getattr(self.heldout, "underpowered", False):
            return "UNDERPOWERED"
        return "NOT CONFIRMED"

    def report(self) -> str:
        L = [f"  generations            {len(self.generations)}",
             f"  candidates scored      {self._k}",
             f"  baseline (train)       {self.base_score:.3f}",
             f"  winner   (train)       {self.best_score:.3f}"
             f"   apparent gain {self.gain:+.3f}"]
        if self.floor is not None:
            rel = "BELOW" if self.gain < self.floor.floor else "above"
            L.append(f"  selection floor        {self.floor.floor:+.3f}"
                     f"   <- the gain is {rel} the floor")
        if self.heldout is not None:
            L += [f"  baseline (held-out)    {self.base_heldout:.3f}",
                  f"  winner   (held-out)    {self.best_heldout:.3f}"
                  f"   real gain {self.best_heldout - self.base_heldout:+.3f}",
                  f"  held-out paired        {self.heldout.wins} fixed /"
                  f" {self.heldout.losses} broken   p={self.heldout.p_value:.4f}"]
        L.append("")
        L.append(f"  verdict: {self.verdict}")
        if self.verdict == "UNDERPOWERED":
            L.append("  (every disagreement favours the winner, but there are"
                     " too few to reach p<0.05)")
        return "\n".join(L)

    _k: int = 0

    def __str__(self) -> str:
        return self.report()


def run_policy(policy: Policy, backend: Backend, tasks: Sequence,
               score: ScoreFn, act=None, route=None) -> Trace:
    """Run one policy over a set of tasks and score every episode.

    A task is `(task_id, state, label)` or a dict with those keys. Anything
    the agent should be able to attribute later, such as the raw input text,
    goes in `state` and is copied into the episode.
    """
    agent = Agent(policy, backend, act=act, route=route)
    tr = Trace()
    for t in tasks:
        if isinstance(t, dict):
            tid, st, lab = t.get("id"), t["state"], t.get("label")
        else:
            tid, st, lab = t[0], t[1], (t[2] if len(t) > 2 else None)
        ep = agent.run(str(tid), st, lab)
        ep.meta["text"] = st.get("text", "")
        ep.outcome = float(score(ep))
        tr.add(ep)
    return tr


def _hits(trace: Trace) -> list[int]:
    return [int(round(e.outcome or 0.0)) for e in trace]


def evolve(policy: Policy, backend: Backend, tasks: Sequence, score: ScoreFn,
           operators, generations: int = 20, candidates: int = 4,
           heldout: Sequence | None = None, act=None, route=None,
           seed: int = 0, verbose: bool = True) -> Result:
    """Evolve a policy, then report how much of the result is selection noise.

    Each generation proposes `candidates` mutants of the current best, scores
    them on `tasks`, and keeps the best if it beats the incumbent. Every
    score ever produced is counted toward the selection floor, including the
    ones that lost: the floor depends on how many candidates were tried, and
    counting only the winners is how a loop convinces itself.

    `heldout` is optional and is the only thing that can make a run credible.
    Without it you get a floor and a warning, which is better than a number
    but is not evidence.
    """
    rng = random.Random(seed)
    base_trace = run_policy(policy, backend, tasks, score, act, route)
    best, best_score = policy, base_trace.score
    gens: list[Generation] = [Generation(0, policy, best_score, "baseline",
                                         True, base_trace)]
    all_scores = [best_score]
    if verbose:
        print(f"  gen  0  {best_score:.3f}  baseline")

    for g in range(1, generations + 1):
        live = [op for op in operators]
        rng.shuffle(live)
        best_cand = None
        for _ in range(candidates):
            cand = None
            for op in live:
                cand = op(best, rng)
                if cand is not None:
                    break
            if cand is None:
                continue
            cand.parent = best.fingerprint()
            tr = run_policy(cand, backend, tasks, score, act, route)
            all_scores.append(tr.score)
            if best_cand is None or tr.score > best_cand[1]:
                best_cand = (cand, tr.score, tr)
        if best_cand is None:
            if verbose:
                print(f"  gen {g:2d}  no operator applied")
            continue
        cand, sc, tr = best_cand
        take = sc > best_score
        gens.append(Generation(g, cand, sc, cand.note, take, tr))
        if verbose:
            print(f"  gen {g:2d}  {sc:.3f}  {cand.note}"
                  f"{'  <- kept' if take else ''}")
        if take:
            best, best_score = cand, sc

    res = Result(gens, policy, best, base_trace.score, best_score)
    res._k = len(all_scores)
    res.floor = _floor(all_scores, len(tasks))
    if heldout:
        b = run_policy(policy, backend, heldout, score, act, route)
        w = run_policy(best, backend, heldout, score, act, route)
        res.base_heldout, res.best_heldout = b.score, w.score
        res.heldout = _confirm(_hits(b), _hits(w))
    return res


def _floor(scores: list[float], n_examples: int):
    """How much a search of this size scores on a policy that never improved.

    Delegated to `evalfloor`, which computes it by simulating the same
    selection procedure over candidates that are all equally good. Splitting
    this out keeps one implementation of the arithmetic rather than two that
    drift.
    """
    try:
        from evalfloor import check
    except ImportError:  # pragma: no cover
        return None
    return check(scores, n_examples=n_examples, baseline=scores[0])


def _confirm(base_hits: list[int], new_hits: list[int]):
    """Paired sign test on held-out data. Also reports when it cannot decide.

    An exact sign test over `d` disagreements cannot return a p-value below
    2^(1-d), so five or fewer can never reach 0.05 no matter how one-sided
    they are. A loop that reports that as a failure will discard real
    improvements for the rest of its life.
    """
    try:
        from evalfloor import confirm
    except ImportError:  # pragma: no cover
        return None
    return confirm(base_hits, new_hits)


def validate_transfer(base: Policy, evolved: Policy, backend: Backend,
                      tasks: Sequence, score: ScoreFn, act=None, route=None):
    """Score a starting schema against an evolved one on a second backend.

    This is the whole "evolve locally, validate hosted" workflow, and it
    costs exactly `2 * len(tasks)` calls rather than the thousands a
    generational loop would spend. Use it after `evolve` has run somewhere
    free.

    No floor is reported and none applies. Nothing is being selected here:
    two fixed policies are scored on the same items and compared pairwise, so
    the sign test is the whole result. Running this over several evolved
    candidates and keeping the best would reintroduce selection, and then the
    floor would apply again.
    """
    a = run_policy(base, backend, tasks, score, act, route)
    b = run_policy(evolved, backend, tasks, score, act, route)
    lat = sorted(d.latency_s for e in list(a) + list(b) for d in e.decisions)
    c = _confirm(_hits(a), _hits(b))
    return Transfer(base_score=a.score, evolved_score=b.score, paired=c,
                    calls=len(tasks) * 2, base_trace=a, evolved_trace=b,
                    p50_s=lat[len(lat) // 2] if lat else 0.0,
                    p90_s=lat[int(len(lat) * 0.9)] if lat else 0.0)


@dataclass
class Transfer:
    """Whether a schema evolved elsewhere survives on this backend."""

    base_score: float
    evolved_score: float
    paired: Any
    calls: int
    p50_s: float = 0.0
    p90_s: float = 0.0
    base_trace: Trace = field(default=None, repr=False)
    evolved_trace: Trace = field(default=None, repr=False)

    @property
    def gain(self) -> float:
        return self.evolved_score - self.base_score

    @property
    def transferred(self) -> bool:
        return bool(self.paired is not None and self.paired.confirmed)

    def report(self) -> str:
        L = [f"  baseline schema        {self.base_score:.3f}",
             f"  evolved schema         {self.evolved_score:.3f}"
             f"   gain {self.gain:+.3f}",
             f"  paired, same items     {self.paired.wins} fixed /"
             f" {self.paired.losses} broken   p={self.paired.p_value:.4f}",
             f"  decision latency       p50 {self.p50_s * 1000:.0f}ms"
             f"   p90 {self.p90_s * 1000:.0f}ms",
             f"  calls used             {self.calls}",
             ""]
        if self.transferred:
            L.append("  TRANSFERRED: the schema is better on this backend too")
        elif getattr(self.paired, "underpowered", False):
            L.append("  UNDERPOWERED: too few disagreements to decide")
        else:
            L.append("  DID NOT TRANSFER: no evidence it is better here")
        return "\n".join(L)

    def __str__(self) -> str:
        return self.report()
