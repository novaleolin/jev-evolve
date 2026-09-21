"""Pick the best ordering on train, or remove the ordering entirely?

The first is a search, so the selection floor applies to it. The second
selects nothing. This is the project's two halves meeting on one number.
"""
import sys
sys.path.insert(0, "/Users/linyang/code/research/jev-evolve")
sys.path.insert(0, "/Users/linyang/code/research/jev-evolve/experiments")
from jev_transfer import base_policy, load, score
from jev_evolve import LocalBackend, Marginalized, run_policy
from jev_evolve.invariance import permutations_of
from evalfloor import check

train, held, _p = load(12, seed=0)
b = LocalBackend()
pol = base_policy()

perms = permutations_of(pol, 8, seed=0)
tr_scores = [run_policy(p, b, train, score).score for p in perms]
best_i = max(range(len(perms)), key=lambda i: tr_scores[i])
print(f"  8 orderings on train: {' '.join(f'{s:.3f}' for s in tr_scores)}")
print(f"  best is ordering {best_i} at {tr_scores[best_i]:.3f}")

c = check(tr_scores, n_examples=len(train), baseline=tr_scores[0])
print(f"  selection floor for 8 candidates on {len(train)} items: "
      f"{c.floor:+.3f}   apparent gain {c.apparent_gain:+.3f}")

held_sel = run_policy(perms[best_i], b, held, score).score
held_base = run_policy(pol, b, held, score).score
held_marg = run_policy(pol, Marginalized(b, 8, seed=0), held, score).score
print(f"\n  held-out, as written            {held_base:.3f}")
print(f"  held-out, best ordering picked  {held_sel:.3f}   {held_sel-held_base:+.3f}")
print(f"  held-out, ordering marginalized {held_marg:.3f}   {held_marg-held_base:+.3f}")
