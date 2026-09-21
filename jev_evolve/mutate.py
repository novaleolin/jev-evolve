"""Mutations: how one generation of a policy becomes the next.

A mutation takes a policy and returns a new one, or None when it does not
apply. `rewrite` is any callable that improves a piece of text: an LLM, a
template, or a hand-written list. Keeping it injected means the loop runs
with no LLM at all, which is what makes an offline generation free.

The operators fall in two groups. The first five edit the policy blind. The
last two read the agent's own trace, which is the only reason to collect one:
a mutation aimed at the option pair the agent actually confused is worth more
than a hundred random rewrites, and it costs one pass over episodes you
already have.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Callable

from .policy import Policy
from .trace import Trace

Mutation = Callable[[Policy, random.Random], "Policy | None"]

#: A rewriter takes (text, slot, rng) and returns a replacement. The rng is
#: part of the contract, not a convenience: a rewriter that reaches for the
#: global `random` makes the whole search unreproducible, and a tool whose
#: business is telling you how much of a gain is real cannot hand you a
#: different answer each time you ask.
Rewriter = Callable[[str, str, random.Random], str]


def mutate_instructions(rewrite: Rewriter) -> Mutation:
    """Rewrite one decision point's instruction text."""
    def op(p: Policy, rng: random.Random):
        names = [n for n, pt in p.points.items()
                 if pt.question.get("instructions")]
        if not names:
            return None
        out = p.copy()
        n = rng.choice(names)
        cur = out.points[n].question["instructions"]
        new = rewrite(cur, f"instructions for {n}", rng)
        if not new or new == cur:
            return None
        out.points[n].question["instructions"] = new
        out.note = f"instructions:{n}"
        return out
    return op


def mutate_criteria(rewrite: Rewriter) -> Mutation:
    """Rewrite the descriptive text of ONE option.

    Per-option rather than wholesale: options compete against each other, so
    the useful edit is usually sharpening the boundary between two of them,
    and a whole-block rewrite makes it impossible to see which edit paid.
    """
    def op(p: Policy, rng: random.Random):
        cands = [(n, k) for n, pt in p.points.items() for k in pt.options]
        if not cands:
            return None
        out = p.copy()
        n, k = rng.choice(cands)
        cur = out.points[n].question["criteria"][k]
        new = rewrite(cur, f"option {k} of {n}", rng)
        if not new or new == cur:
            return None
        out.points[n].question["criteria"][k] = new
        out.note = f"criteria:{n}.{k}"
        return out
    return op


def mutate_threshold(grid=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                           0.9)) -> Mutation:
    """Move one confidence threshold. Cheap, and often worth more than text.

    The grid reaches down to 0.05 because a model's absolute probabilities
    are only meaningful relative to its own calibration, and one that puts
    0.15 on everything it likes is not broken, only differently calibrated. A
    grid starting at 0.1 reports such a model as hopeless.
    """
    def op(p: Policy, rng: random.Random):
        if not p.points:
            return None
        out = p.copy()
        n = rng.choice(list(out.points))
        cur = out.points[n].threshold
        opts = [g for g in grid if g != cur]
        out.points[n].threshold = rng.choice(opts)
        out.note = f"threshold:{n}={out.points[n].threshold}"
        return out
    return op


def mutate_state_fields(available: list[str]) -> Mutation:
    """Add or drop one field from what a decision point can see.

    This is the operator with the largest measured effect in our own prior
    work, and the one most often settled by assumption. A field judged
    "irrelevant to a decision model" and dropped by hand turned out, once
    measured, to be worth nine accuracy points. Vendors warn that a large
    irrelevant state costs accuracy, which makes dropping feel safe; whether
    a specific field is irrelevant is an empirical question, not a stylistic
    one, so it belongs in the search rather than in a design review.
    """
    def op(p: Policy, rng: random.Random):
        if not p.points:
            return None
        out = p.copy()
        n = rng.choice(list(out.points))
        pt = out.points[n]
        cur = list(pt.state_fields if pt.state_fields is not None else available)
        missing = [f for f in available if f not in cur]
        if missing and (not cur or rng.random() < 0.5):
            f = rng.choice(missing)
            cur.append(f)
            out.note = f"state:{n}+{f}"
        elif len(cur) > 1:
            f = rng.choice(cur)
            cur.remove(f)
            out.note = f"state:{n}-{f}"
        else:
            return None
        pt.state_fields = cur
        return out
    return op


def mutate_order() -> Mutation:
    """Swap two decision points.

    Order is not cosmetic in an agent: an earlier point writes into the state
    the later ones read, so moving a cheap filter ahead of an expensive
    discrimination removes most of the options the second one has to tell
    apart.
    """
    def op(p: Policy, rng: random.Random):
        if len(p.order) < 2:
            return None
        out = p.copy()
        i, j = rng.sample(range(len(out.order)), 2)
        out.order[i], out.order[j] = out.order[j], out.order[i]
        out.note = f"order:{out.order[i]}<->{out.order[j]}"
        return out
    return op


def mutate_option_order() -> Mutation:
    """Permute the options of one choice point.

    Measured worth on an 8-option task: the same schema scored 0.260 under
    one ordering and 0.521 averaged over two, while every text operator in
    this module found nothing at all on the same run. Option order is not a
    presentation detail, it is the largest single dimension of the search,
    and it was missing.

    Searching over it is the cheap response. `Marginalized` is the better
    one, because it removes the dependence rather than finding the ordering
    that happens to exploit it, and an ordering selected on a training split
    is exactly the kind of gain the selection floor exists to catch.
    """
    def op(p: Policy, rng: random.Random):
        live = [n for n, pt in p.points.items() if len(pt.options) > 1]
        if not live:
            return None
        out = p.copy()
        n = rng.choice(live)
        crit = out.points[n].question["criteria"]
        keys = list(crit)
        shuffled = keys[:]
        for _ in range(8):
            rng.shuffle(shuffled)
            if shuffled != keys:
                break
        else:
            return None
        out.points[n].question["criteria"] = {k: crit[k] for k in shuffled}
        out.note = f"order:{n} options permuted"
        return out
    return op


def mutate_criteria_from_examples(examples_by_label, k=2) -> Mutation:
    """Describe an option using real inputs that belong to it.

    The description anyone writes first is the label name again, which tells
    the model nothing the label did not already say and leaves neighbouring
    options indistinguishable. Two real examples usually separate them.

    Needs no LLM, which is the point: an offline generation stays free, so
    the cost of checking a policy is never a reason to skip the check.
    """
    def op(p: Policy, rng: random.Random):
        cands = [(n, lab) for n, pt in p.points.items() for lab in pt.options
                 if examples_by_label.get(lab)]
        if not cands:
            return None
        out = p.copy()
        n, lab = rng.choice(cands)
        pool = examples_by_label[lab]
        cur = out.points[n].question["criteria"][lab]
        fresh = [x for x in sorted(set(pool)) if f'"{x}"' not in cur]
        if not fresh:
            return None
        picks = rng.sample(fresh, min(k, len(fresh)))
        add = " | ".join(f'"{x}"' for x in picks)
        # Append. An earlier version replaced the criteria outright, which is
        # harmless when the starting text is the label name repeated back but
        # destroys a real hand-written rule, and in a measured run every such
        # candidate scored below the baseline it was mutating.
        new = (cur + " Examples: " if cur else "Examples: ") + add
        out.points[n].question["criteria"][lab] = new
        out.note = f"examples:{lab}"
        return out
    return op


# ------------------------------------------------------ trace-driven operators
#
# These are the reason the agent records anything. Everything above edits the
# policy without knowing how the agent did; everything below aims the edit at
# a failure the agent actually made.


def mutate_from_confusions(rewrite: Rewriter, trace: Trace, gold=None,
                           top: int = 5) -> Mutation:
    """Rewrite the option the agent most often picks by mistake.

    Sampled from the most frequent confusions rather than always taking the
    top one, because a search that fixes the same pair every generation stops
    exploring after the first fix that does not take.

    The rewriter is told what the option is being confused WITH. That is the
    whole content of the edit: an option's description is not wrong in
    isolation, it is wrong at a boundary, and a rewriter given only the
    option text will produce a better paraphrase of the same mistake.
    """
    from .analyze import confusions, default_gold
    pairs = confusions(trace, gold or default_gold).most_common(top)

    def op(p: Policy, rng: random.Random):
        live = [(n, pred, act, c) for (n, pred, act), c in pairs
                if n in p.points and pred in p.points[n].options]
        if not live:
            return None
        n, pred, act, _c = rng.choices(live, weights=[r[3] for r in live])[0]
        out = p.copy()
        cur = out.points[n].question["criteria"][pred]
        new = rewrite(cur, f"option {pred} of {n}, wrongly chosen over {act}", rng)
        if not new or new == cur:
            return None
        out.points[n].question["criteria"][pred] = new
        out.note = f"confusion:{pred}>{act}"
        return out
    return op


def mutate_criteria_from_errors(trace: Trace, gold=None, k: int = 2) -> Mutation:
    """Describe an option using inputs the agent got WRONG for it.

    The same construction as `mutate_criteria_from_examples`, drawing from a
    different pool: the episodes whose true answer was this option and which
    the agent missed. Those inputs are, by construction, the ones the current
    description fails to cover, so they carry more information per token than
    examples drawn at random from the training set.

    Needs no LLM. An agent with a trace can improve its own policy offline,
    which is the cheapest form of self-improvement available and the one
    nobody runs, because collecting the trace is the part people skip.
    """
    from .analyze import default_gold
    g = gold or default_gold
    pool: dict = {}
    for ep in trace:
        truth = g(ep)
        for d in ep.decisions:
            want = truth.get(d.question)
            if want is None or d.choice == want:
                continue
            text = ep.meta.get("text") or ep.task_id
            pool.setdefault((d.question, want), []).append(str(text))

    def op(p: Policy, rng: random.Random):
        live = [key for key in pool
                if key[0] in p.points and key[1] in p.points[key[0]].options]
        if not live:
            return None
        n, lab = rng.choice(live)
        out = p.copy()
        cur = out.points[n].question["criteria"][lab]
        # sorted(set(...)) rather than set(...): set iteration order over
        # strings depends on per-process hash randomisation, so sampling
        # straight from a set makes two identical runs disagree in different
        # processes. A within-process reproducibility test cannot catch that.
        seen = sorted(set(pool[(n, lab)]))
        fresh = [x for x in seen if f'"{x}"' not in cur]
        if not fresh:
            return None
        picks = rng.sample(fresh, min(k, len(fresh)))
        add = " | ".join(f'"{x}"' for x in picks)
        if "covers:" in cur.lower():
            new = cur + " | " + add
        else:
            new = (cur + " Also covers: " if cur else "Covers: ") + add
        if new == cur:
            return None
        out.points[n].question["criteria"][lab] = new
        out.note = f"missed:{lab}"
        return out
    return op


def mutate_threshold_from_trace(trace: Trace, n_levels: int = 8) -> Mutation:
    """Move a threshold, with the grid read off what the backend actually says.

    The fixed grid in `mutate_threshold` is wrong for most backends, and it is
    wrong in both directions at once. Measured on an 8-option task, a 0.5B
    local model put its top probability between 0.28 and 0.59 for the middle
    eighty percent of decisions. Against that, every grid value at or below
    0.2 was the same no-op, and every value at or above 0.5 abstained on four
    decisions in five and scored near zero. Two of eleven grid points did
    anything at all, and in a 15-generation run a third of the candidate
    slots went to values that could not have helped.

    The fix is not a better constant. It is to build the grid from the
    quantiles of the top probabilities the backend produced on this task, so
    an 8-way choice and a 2-way noul get different grids, as they should, and
    a differently calibrated model gets a grid that fits it.
    """
    tops: list[dict] = []
    for ep in trace:
        for d in ep.decisions:
            if d.probabilities:
                tops.append({"point": d.question,
                             "p": max(d.probabilities.values())})
    by: dict = {}
    for row in tops:
        by.setdefault(row["point"], []).append(row["p"])
    grids = {}
    for point, ps in by.items():
        ps.sort()
        qs = sorted({round(ps[min(len(ps) - 1, int(len(ps) * (i + 1)
                                                  / (n_levels + 1)))], 3)
                     for i in range(n_levels)})
        grids[point] = [0.0] + qs

    def op(p: Policy, rng: random.Random):
        live = [n for n in p.points if grids.get(n)]
        if not live:
            return None
        n = rng.choice(live)
        out = p.copy()
        opts = [g for g in grids[n] if g != out.points[n].threshold]
        if not opts:
            return None
        out.points[n].threshold = rng.choice(opts)
        out.note = f"threshold:{n}={out.points[n].threshold}"
        return out
    return op


def default_operators(available_fields: list[str] | None = None,
                      examples_by_label: dict | None = None,
                      trace: Trace | None = None,
                      rewrite: Rewriter | None = None) -> list[Mutation]:
    """A working set of operators from whatever you happen to have.

    Every argument is optional and each one that is present adds operators.
    With nothing but a policy you still get threshold and order search, which
    runs offline, needs no LLM, and is frequently where the gain is.
    """
    ops: list[Mutation] = [mutate_order(), mutate_option_order()]
    ops.append(mutate_threshold_from_trace(trace) if trace is not None
               and len(trace) else mutate_threshold())
    if available_fields:
        ops.append(mutate_state_fields(available_fields))
    if examples_by_label:
        ops.append(mutate_criteria_from_examples(examples_by_label))
    if trace is not None and len(trace):
        ops.append(mutate_criteria_from_errors(trace))
        if rewrite is not None:
            ops.append(mutate_from_confusions(rewrite, trace))
    if rewrite is not None:
        ops += [mutate_instructions(rewrite), mutate_criteria(rewrite)]
    return ops
