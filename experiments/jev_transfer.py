#!/usr/bin/env python3
"""Does a schema evolved against a small local model transfer to a hosted one?

This is the experiment the project rests on. The README tells you to evolve
against something you own and validate the winner against the hosted
endpoint, both to stay inside a provider's terms and because a generational
loop is thousands of decisions. That advice is worthless if the schema a
0.5B model likes is not the schema the hosted model likes.

Stage 1 (offline, free) evolves the schema against Qwen2.5-0.5B.
Stage 2 (hosted, ~2N calls) scores ONLY the starting schema and the winner.

The starting schema is the same eight-option rule set used in the earlier
Jev-versus-cheap-LLM comparison, so the numbers here attach to that result:
Jev scored 0.856 on it, against 0.900 for the best cheap generative model.
The question is whether that gap is a property of the model or of the schema.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jev_evolve import Policy, choice, evolve, run_policy
from jev_evolve.mutate import (mutate_criteria_from_errors,
                               mutate_criteria_from_examples,
                               mutate_state_fields, mutate_threshold)

INTENTS = ["card_arrival", "card_delivery_estimate", "card_not_working",
           "card_payment_fee_charged", "declined_card_payment",
           "lost_or_stolen_card", "pending_card_payment", "card_swallowed"]

#: The starting schema. One written rule per option, the way a careful
#: engineer writes it on day one and then never measures.
RULES = {
 "card_arrival": "The customer is asking where a card they ordered is, or when it will arrive.",
 "card_delivery_estimate": "The customer asks how long delivery takes in general, not about one specific order.",
 "card_not_working": "A card the customer already holds is being rejected or will not work.",
 "card_payment_fee_charged": "The customer was charged a fee on a card payment and is asking why.",
 "declined_card_payment": "A specific card payment was declined and the customer wants to know the reason.",
 "lost_or_stolen_card": "The customer reports a card lost or stolen, or suspects theft.",
 "pending_card_payment": "A card payment is showing as pending or has not settled.",
 "card_swallowed": "An ATM or machine physically retained the customer's card.",
}

INSTRUCTIONS = "Which intent matches the customer message?"


def load(n_per_intent: int, seed: int = 0):
    from datasets import load_dataset
    ds = load_dataset("legacy-datasets/banking77", split="test")
    names = ds.features["label"].names
    keep = {names.index(i) for i in INTENTS}
    by: dict = {}
    for r in ds:
        if r["label"] in keep:
            by.setdefault(names[r["label"]], []).append(r["text"])
    rng = random.Random(seed)
    train, held, pool = [], [], {}
    for lab, texts in by.items():
        rng.shuffle(texts)
        take = texts[:n_per_intent * 2]
        train += [(t, lab) for t in take[:n_per_intent]]
        held += [(t, lab) for t in take[n_per_intent:n_per_intent * 2]]
        pool[lab] = take[:n_per_intent]
    rng.shuffle(train)
    rng.shuffle(held)
    fmt = lambda xs: [(f"i{i}", {"customer_message": t}, lab)
                      for i, (t, lab) in enumerate(xs)]
    return fmt(train), fmt(held), pool


def score(ep):
    return float(ep.decisions[0].choice == ep.label)


def base_policy():
    return Policy({"intent": choice(INSTRUCTIONS, dict(RULES))})


def stage1_offline(args):
    """Evolve against a local 0.5B model. No API key, no network."""
    from jev_evolve import LocalBackend
    train, held, pool = load(args.per_intent, seed=args.seed)
    backend = LocalBackend(model=args.local_model)
    print(f"  local backend  {args.local_model} on {backend.device}")
    print(f"  {len(train)} train / {len(held)} held-out, {len(INTENTS)} options\n")

    policy = base_policy()
    t0 = time.time()
    trace = run_policy(policy, backend, train, score)
    print(f"  starting accuracy  {trace.score:.3f}")
    for (pt, pred, act), c in __import__("jev_evolve").confusions(trace).most_common(3):
        print(f"    confuses {pred} for {act}  x{c}")
    print()

    ops = [mutate_criteria_from_errors(trace),
           mutate_criteria_from_examples(pool),
           mutate_threshold(),
           mutate_state_fields(["customer_message"])]
    res = evolve(policy, backend, train, score, ops,
                 generations=args.generations, candidates=args.candidates,
                 heldout=held, seed=args.seed)
    print()
    print(res.report())
    print(f"\n  offline wall clock  {time.time() - t0:.0f}s, 0 API calls")

    res.best.save(args.out)
    print(f"  evolved schema -> {args.out}")
    return res


def stage2_hosted(args):
    """Score the starting schema and the winner on the hosted endpoint."""
    from jev_evolve import JevBackend, validate_transfer
    _train, held, _pool = load(args.per_intent, seed=args.seed)
    held = held[:args.hosted_n]
    t = validate_transfer(base_policy(), Policy.load(args.out),
                          JevBackend(model=args.jev_model), held, score)
    print(t.report())
    with open(args.results, "w") as f:
        json.dump({"model": args.jev_model, "n": len(held),
                   "baseline": t.base_score, "evolved": t.evolved_score,
                   "wins": t.paired.wins, "losses": t.paired.losses,
                   "p": t.paired.p_value, "calls": t.calls,
                   "p50_ms": round(t.p50_s * 1000), "p90_ms": round(t.p90_s * 1000)},
                  f, indent=2)
    print(f"  -> {args.results}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["offline", "hosted"])
    ap.add_argument("--per-intent", type=int, default=12)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--hosted-n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--jev-model", default="typesafe/jev-1.13")
    ap.add_argument("--out", default="experiments/evolved_schema.json")
    ap.add_argument("--results", default="experiments/transfer_results.json")
    a = ap.parse_args()
    (stage1_offline if a.stage == "offline" else stage2_hosted)(a)
