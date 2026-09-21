#!/usr/bin/env python3
"""Does order dependence grow with the number of options?

Practitioners report that a typed decision model's label boundaries get
unstable as the candidate set grows, while small candidate sets stay
reliable. If that instability is partly the option ordering, then the number
of options predicts how much of the answer is formatting, and there is an
actionable line: above some count, marginalize rather than tune.

Same model, same items, same rules; only the size of the option set changes.
Items are restricted to the labels in each subset so the task stays honest at
every size.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jev_transfer import INSTRUCTIONS, INTENTS, RULES, load, score

from jev_evolve import (LocalBackend, Marginalized, Policy, choice,
                        permutation_sensitivity, run_policy)


def main(args):
    train, _held, _pool = load(args.per_intent, seed=args.seed)
    backend = LocalBackend(model=args.local_model)
    print(f"  {args.local_model}\n")
    print(f"  options   n   excess flip   acc spread   as written   "
          f"marginalized k={args.k}")
    rows = []
    for m in args.sizes:
        keep = INTENTS[:m]
        tasks = [t for t in train if t[2] in keep]
        pol = Policy({"intent": choice(INSTRUCTIONS,
                                       {k: RULES[k] for k in keep})})
        s = permutation_sensitivity(pol, backend, tasks, score, k=args.k,
                                    seed=args.seed)
        plain = run_policy(pol, backend, tasks, score).score
        marg = run_policy(pol, Marginalized(backend, args.k, seed=args.seed),
                          tasks, score).score
        rows.append({"options": m, "n": len(tasks),
                     "excess_flip_rate": s.excess_flip_rate,
                     "spread": s.spread, "control_spread": s.control_spread,
                     "as_written": plain, "marginalized": marg})
        print(f"  {m:7d} {len(tasks):4d}   {s.excess_flip_rate:10.1%}"
              f"   {s.spread:10.3f}   {plain:10.3f}   {marg:8.3f}"
              f"  {marg - plain:+.3f}")

    with open(args.results, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  -> {args.results}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--per-intent", type=int, default=12)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--results", default="experiments/option_count_results.json")
    main(ap.parse_args())
