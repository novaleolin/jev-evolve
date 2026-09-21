#!/usr/bin/env python3
"""How much of a typed decision's answer is the option order?

Three measurements on the same 8-option banking task, with the same model
and the same hand-written rules:

  1. the sensitivity check, against a fixed-order control
  2. accuracy under single orderings, which is what you ship
  3. accuracy under `Marginalized`, which averages the ordering away

This is the experiment that redirected the project. A first attempt evolved
the criteria text of this exact schema for 14 generations and 43 candidates
and improved nothing, because the thing deciding the answer was not in the
text being mutated.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jev_transfer import INSTRUCTIONS, RULES, base_policy, load, score

from jev_evolve import Marginalized, permutation_sensitivity, run_policy
from jev_evolve.invariance import permutations_of


def main(args):
    train, held, _pool = load(args.per_intent, seed=args.seed)
    if args.backend == "local":
        from jev_evolve import LocalBackend
        backend = LocalBackend(model=args.local_model)
        label = args.local_model
    else:
        from jev_evolve import JevBackend
        backend = JevBackend(model=args.jev_model)
        label = args.jev_model
    print(f"  backend {label}   {len(train)} train / {len(held)} held-out"
          f"   {len(RULES)} options\n")

    pol = base_policy()
    print("1. is the answer decided by option order?\n")
    s = permutation_sensitivity(pol, backend, train, score, k=args.k,
                                seed=args.seed)
    print(s.report())

    print("\n2. accuracy under individual orderings (what you ship)\n")
    accs = []
    for i, p in enumerate(permutations_of(pol, args.k, seed=args.seed)):
        a = run_policy(p, backend, train, score).score
        accs.append(a)
        print(f"    ordering {i}{' (as written)' if i == 0 else ''}   {a:.3f}")
    print(f"    worst {min(accs):.3f}   best {max(accs):.3f}"
          f"   spread {max(accs) - min(accs):.3f}")

    print("\n3. accuracy with the ordering averaged away\n")
    rows = []
    for split, tasks in (("train", train), ("held-out", held)):
        base = run_policy(pol, backend, tasks, score).score
        marg = run_policy(pol, Marginalized(backend, args.k, seed=args.seed),
                          tasks, score).score
        rows.append({"split": split, "as_written": base,
                     "marginalized_k": args.k, "marginalized": marg})
        print(f"    {split:9s} as written {base:.3f}"
              f"   marginalized k={args.k} {marg:.3f}"
              f"   {marg - base:+.3f}")

    out = {"backend": label, "k": args.k, "orderings": accs,
           "flip_rate": s.flip_rate, "noise_rate": s.noise_rate,
           "excess_flip_rate": s.excess_flip_rate, "splits": rows}
    with open(args.results, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  -> {args.results}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "jev"], default="local")
    ap.add_argument("--per-intent", type=int, default=12)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--jev-model", default="typesafe/jev-1.13")
    ap.add_argument("--results", default="experiments/order_results.json")
    main(ap.parse_args())
