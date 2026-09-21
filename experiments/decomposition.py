#!/usr/bin/env python3
"""Three ways to stop a resolved past event from capturing the decision.

The probe in salience_probe.py shows the failure. This asks what fixes it,
and in an order that can embarrass the expensive answer:

  A  baseline           one 8-way choice, as written
  B  instruction fix    the same one question, told to ignore resolved matters
  C  decomposition      a separate typed question asking whether a resolved
                        prior matter is present, routing to an intent
                        question that knows the answer

B is tested before C on purpose. Decomposition costs a second decision per
episode and is the interesting idea; a one-line instruction change costs
nothing. If B closes the gap, C is not worth having, and this run says so.

Ordering is marginalized throughout, because the same schema's answers move
under permutation and a content effect read off that would be noise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jev_transfer import INSTRUCTIONS, RULES, base_policy, load, score
from salience_probe import build

from jev_evolve import Agent, Marginalized, Policy, choice, noul, run_policy

IGNORE_RESOLVED = (
    "Route the request the customer is making NOW. If the message also "
    "mentions an earlier matter that the customer says is already resolved, "
    "closed or settled, that earlier matter is not the request: ignore it.")


def arm_b():
    return Policy({"intent": choice(f"{INSTRUCTIONS} {IGNORE_RESOLVED}",
                                    dict(RULES))})


def arm_c():
    """A real two-decision agent: gate first, then the intent question the
    gate selects. `route` is what makes the second question depend on the
    first, which is the whole content of a decomposition."""
    pol = Policy({
        "has_resolved_prior": noul(
            "Does this message quote or describe an earlier matter that the "
            "customer says is already resolved, closed or settled?",
            threshold=0.5),
        "intent_plain": choice(INSTRUCTIONS, dict(RULES)),
        "intent_after_prior": choice(f"{INSTRUCTIONS} {IGNORE_RESOLVED}",
                                     dict(RULES)),
    }, order=["has_resolved_prior"])

    def route(state, asked):
        if "has_resolved_prior" not in asked:
            return "has_resolved_prior"
        if len(asked) > 1:
            return None
        return ("intent_after_prior" if state.get("resolved_prior")
                else "intent_plain")

    def act(state, answer):
        if answer.name == "has_resolved_prior":
            return {"resolved_prior": bool(answer)}
        return {"intent": answer.choice}

    return pol, route, act


def score_last_intent(ep):
    """Score whichever intent question the route actually reached."""
    picks = [d for d in ep.decisions if d.probabilities]
    return float(bool(picks) and picks[-1].choice == ep.label)


def capture_rate(trace, distractor):
    """Share of wrong answers that landed on the resolved prior event."""
    wrong = to_d = 0
    for ep, d in zip(trace, distractor):
        picks = [x for x in ep.decisions if x.probabilities]
        pick = picks[-1].choice if picks else None
        if pick != ep.label:
            wrong += 1
            to_d += int(pick == d)
    return wrong, to_d, (to_d / wrong if wrong else 0.0)


def main(args):
    train, held, pool = load(args.per_intent, seed=args.seed)
    base, pert, distractor = build(train + held, pool, seed=args.seed)

    from jev_evolve import LocalBackend
    backend = Marginalized(LocalBackend(model=args.local_model), args.k,
                           seed=args.seed)
    print(f"  {args.local_model}   {len(pert)} perturbed items"
          f"   ordering marginalized k={args.k}\n")

    rows = []

    def record(name, trace, decisions_per_item):
        w, td, share = capture_rate(list(trace), distractor)
        rows.append({"arm": name, "accuracy": trace.score, "wrong": w,
                     "to_distractor": td, "capture_share": share,
                     "decisions_per_item": decisions_per_item})
        print(f"  {name:28s} acc {trace.score:.3f}"
              f"   captured {td}/{w} ({share:.1%})"
              f"   {decisions_per_item:.2f} decisions/item")

    record("A baseline", run_policy(base_policy(), backend, pert, score), 1.0)
    record("B instruction fix", run_policy(arm_b(), backend, pert, score), 1.0)

    pol, route, act = arm_c()
    from jev_evolve import Trace
    agent = Agent(pol, backend, act=act, route=route)
    tr = Trace()
    for tid, st, lab in pert:
        ep = agent.run(str(tid), st, lab)
        ep.outcome = score_last_intent(ep)
        tr.add(ep)
    record("C decomposition", tr,
           sum(e.n_decisions for e in tr) / len(tr))

    # What the gate itself got right, since a decomposition is only as good
    # as its first question and a silent 50% gate would look like a wash.
    gate_yes = sum(int(bool(e.decisions[0].noul and e.decisions[0].noul >= 0.5))
                   for e in tr)
    print(f"\n  gate said 'a resolved prior is present' on {gate_yes}/{len(tr)}"
          f" items; every perturbed item has one")

    with open(args.results, "w") as f:
        json.dump({"k": args.k, "n": len(pert), "gate_yes": gate_yes,
                   "arms": rows}, f, indent=2)
    print(f"\n  -> {args.results}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-intent", type=int, default=12)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--results", default="experiments/decomposition_results.json")
    main(ap.parse_args())
