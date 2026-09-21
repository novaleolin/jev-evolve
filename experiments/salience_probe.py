#!/usr/bin/env python3
"""Does the decision grab the salient past event instead of the live request?

A practitioner report: these models get pulled toward the final or most vivid
phenomenon and miss the responsible party, the root cause or the
precondition. That is a claim about content, not formatting, so it needs a
different probe from the ordering checks.

The construction is a contrast set (Gardner et al., 2020). Every item is a
real banking77 utterance of intent A. The perturbed twin prepends a real
utterance of a different intent B, marked as finished and handled, and the
gold label stays A: what happened last month and was resolved does not change
where today's request should be routed.

The discriminator is the point. If prepending text merely degrades the
decision, the new errors spread across labels. If the model is captured by
the salient prior event, the errors concentrate on B specifically. Those two
make different predictions and the run reports both.

Order dependence is fixed first, with Marginalized, because a schema whose
answers move under permutation cannot be used to measure anything else.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jev_transfer import INSTRUCTIONS, INTENTS, RULES, base_policy, load, score

from jev_evolve import Marginalized, Policy, choice, run_policy

#: Each frame quotes the prior message and marks it finished, so the live
#: request is the only thing left to route. Quoting matters: banking77
#: utterances are whole sentences with their own punctuation and often their
#: own question marks, and splicing one into a clause produced sentences that
#: contradicted the frame ("...now that the bank is closed?, which your team
#: closed"). The wording varies because a single template measures one
#: sentence rather than the phenomenon.
FRAMES = [
    'Last month I wrote: "{b}" That was resolved. Today I need something else: {a}',
    'Earlier I asked you: "{b}" Your team closed that one. Separate request: {a}',
    'A while back my message was: "{b}" It is fully handled now. New question: {a}',
    'Previously I contacted you about this: "{b}" Settled since. Unrelated: {a}',
]


def build(tasks, pool, seed=0):
    """Pair every item with a twin carrying a resolved distractor clause."""
    rng = random.Random(seed)
    base, pert, distractor = [], [], []
    for tid, st, lab in tasks:
        others = [x for x in INTENTS if x != lab and pool.get(x)]
        if not others:
            continue
        b_lab = rng.choice(others)
        b_text = rng.choice(pool[b_lab])
        frame = rng.choice(FRAMES)
        base.append((tid, dict(st), lab))
        pert.append((tid, dict(st, customer_message=frame.format(
            b=b_text.strip(), a=st["customer_message"])), lab))
        distractor.append(b_lab)
    return base, pert, distractor


def analyse(base_trace, pert_trace, distractor, labels):
    """Split the damage into capture by the distractor and everything else.

    Two controls, because 1/(m-1) is a theoretical chance level and not this
    model's behaviour. The matched control is the one that decides: the
    distractor label is drawn at random and independently of the model, so
    how often the model names it on the UNPERTURBED items, among the ones it
    already gets wrong, is its own baseline rate for that label. Capture
    means the perturbed rate stands well above that, not merely above 1/(m-1).
    """
    m = len(labels)
    chance = 1.0 / (m - 1)
    newly_wrong, to_distractor, elsewhere, fixed = 0, 0, 0, 0
    base_wrong, base_wrong_to_distractor = 0, 0
    for b, p, d in zip(base_trace, pert_trace, distractor):
        gold = b.label
        b_pick = b.decisions[0].choice
        p_pick = p.decisions[0].choice
        if b_pick != gold:
            base_wrong += 1
            base_wrong_to_distractor += int(b_pick == d)
        if b_pick == gold and p_pick != gold:
            newly_wrong += 1
            to_distractor += int(p_pick == d)
            elsewhere += int(p_pick != d)
        elif b_pick != gold and p_pick == gold:
            fixed += 1
    share = to_distractor / newly_wrong if newly_wrong else 0.0
    matched = (base_wrong_to_distractor / base_wrong) if base_wrong else 0.0

    # One-sided exact binomial tail against the matched control.
    from math import comb
    q = matched if matched > 0 else chance
    pval = sum(comb(newly_wrong, i) * q ** i * (1 - q) ** (newly_wrong - i)
               for i in range(to_distractor, newly_wrong + 1)) \
        if newly_wrong else 1.0
    return {"options": m, "newly_wrong": newly_wrong, "fixed": fixed,
            "to_distractor": to_distractor, "elsewhere": elsewhere,
            "share_to_distractor": share, "expected_if_random": chance,
            "matched_control": matched, "base_wrong": base_wrong,
            "ratio_vs_matched": (share / matched) if matched else None,
            "p_value": pval}


def main(args):
    train, held, pool = load(args.per_intent, seed=args.seed)
    tasks = train + held
    base, pert, distractor = build(tasks, pool, seed=args.seed)

    from jev_evolve import LocalBackend
    inner = LocalBackend(model=args.local_model)
    # Order first. A schema that moves under permutation cannot measure a
    # content effect, because the content effect would be read off noise.
    backend = inner if args.k <= 1 else Marginalized(inner, args.k,
                                                     seed=args.seed)
    pol = base_policy()
    print(f"  {args.local_model}   {len(base)} pairs   "
          f"{'ordering marginalized k=%d' % args.k if args.k > 1 else 'raw'}\n")

    b = run_policy(pol, backend, base, score)
    p = run_policy(pol, backend, pert, score)
    print(f"  accuracy, live request only          {b.score:.3f}")
    print(f"  accuracy, resolved clause prepended  {p.score:.3f}"
          f"   ({p.score - b.score:+.3f})")

    a = analyse(list(b), list(p), distractor, list(RULES))
    print(f"\n  of the {a['newly_wrong']} answers the clause broke:")
    print(f"    went to the resolved event   {a['to_distractor']:3d}"
          f"   ({a['share_to_distractor']:.1%})")
    print(f"    went somewhere else          {a['elsewhere']:3d}")
    print(f"\n  what that rate is measured against:")
    print(f"    uniform over wrong labels    {a['expected_if_random']:.1%}")
    print(f"    this model's own rate for    {a['matched_control']:.1%}"
          f"   ({a['base_wrong']} unperturbed errors)")
    if a["ratio_vs_matched"]:
        print(f"    -> {a['ratio_vs_matched']:.1f}x the matched control"
              f"   p={a['p_value']:.2e}")
    print(f"  ({a['fixed']} answers the clause happened to fix)")

    picks = Counter(e.decisions[0].choice for e in p)
    print(f"\n  predicted-label spread under perturbation: "
          f"{len(picks)}/{len(RULES)} labels used")

    with open(args.results, "w") as f:
        json.dump({"backend": args.local_model, "k": args.k,
                   "pairs": len(base), "base": b.score, "perturbed": p.score,
                   **a}, f, indent=2)
    print(f"\n  -> {args.results}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-intent", type=int, default=12)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--results", default="experiments/salience_results.json")
    main(ap.parse_args())
