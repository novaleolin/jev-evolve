"""Reading an agent's own trace.

The loop needs to know more than "this generation scored 0.81". It needs to
know which decision point lost the episodes, which two options were swapped
for each other, and which decisions were confident and wrong rather than
merely uncertain. Those three questions are what turn a random search into a
targeted one.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

from .trace import Episode, Trace

#: Maps an episode to the answer each decision point should have given.
#: Returning {} means "no ground truth for this episode", which is normal:
#: most agents can score an outcome long before they can attribute it.
GoldFn = Callable[[Episode], dict]


def default_gold(ep: Episode) -> dict:
    """Attribute a labelled episode to its single choice point.

    Only fires when there is exactly one choice decision, because with two
    the attribution is a guess, and a guessed attribution sends every later
    mutation at the wrong point.
    """
    if ep.label is None:
        return {}
    picks = [d for d in ep.decisions if d.probabilities]
    if len(picks) != 1:
        return {}
    return {picks[0].question: ep.label}


def confusions(trace: Trace, gold: GoldFn = default_gold) -> Counter:
    """Count (point, predicted, actual) triples the agent got wrong.

    Sorted by frequency, this is a to-do list. The top entry is the option
    pair whose written boundary is costing the most, and it is the pair the
    next generation should spend its mutations on.
    """
    out: Counter = Counter()
    for ep in trace:
        g = gold(ep)
        if not g:
            continue
        for d in ep.decisions:
            want = g.get(d.question)
            if want is None or d.choice is None or d.choice == want:
                continue
            out[(d.question, d.choice, want)] += 1
    return out


def point_accuracy(trace: Trace, gold: GoldFn = default_gold) -> dict:
    """Per-decision-point accuracy and abstention rate.

    An agent's headline score averages over points that are already solved
    and points that are hopeless. This separates them, which is usually
    enough to see that one point owns most of the loss.
    """
    hit: dict = defaultdict(int)
    tot: dict = defaultdict(int)
    absten: dict = defaultdict(int)
    for ep in trace:
        g = gold(ep)
        for d in ep.decisions:
            if d.choice is None:
                absten[d.question] += 1
            want = g.get(d.question)
            if want is None:
                continue
            tot[d.question] += 1
            hit[d.question] += int(d.choice == want)
    return {n: {"n": tot[n], "accuracy": hit[n] / tot[n] if tot[n] else None,
                "abstentions": absten[n]}
            for n in set(tot) | set(absten)}


def overconfident(trace: Trace, gold: GoldFn = default_gold,
                  margin: float = 0.5) -> list:
    """Wrong decisions the agent was sure about.

    These matter more than ordinary errors twice over. They are the ones a
    confidence threshold cannot catch, so no amount of threshold tuning will
    help; and they usually mean an option's written criteria positively
    describe the wrong case, rather than merely failing to describe the right
    one. A rewrite is the only fix, and this is the list to rewrite from.
    """
    out = []
    for ep in trace:
        g = gold(ep)
        for d in ep.decisions:
            want = g.get(d.question)
            if want is None or d.choice is None or d.choice == want:
                continue
            if d.margin >= margin:
                out.append((ep.task_id, d.question, d.choice, want, d.margin))
    return sorted(out, key=lambda r: -r[-1])


def cost(trace: Trace) -> dict:
    """Decisions and seconds per episode.

    A generation that buys two accuracy points by asking three more questions
    has not obviously won, and a loop that only reports accuracy will keep
    making that trade until the agent is slower than the model it replaced.
    """
    n = len(trace) or 1
    decs = sum(e.n_decisions for e in trace)
    secs = sum(e.total_latency for e in trace)
    lat = sorted(d.latency_s for e in trace for d in e.decisions)
    return {"episodes": len(trace), "decisions_per_episode": decs / n,
            "seconds_per_episode": secs / n,
            "p50_decision_s": lat[len(lat) // 2] if lat else 0.0,
            "p90_decision_s": lat[int(len(lat) * 0.9)] if lat else 0.0}
