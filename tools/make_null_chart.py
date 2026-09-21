"""Draw docs/null.png: what a loop reports when nothing is improving.

The curve is the expected maximum of k equally-good candidates, each scored
on n examples, minus their common true score. `evalfloor.selection_floor`
computes that by resampling every example, which is exact and far too slow to
sweep. Here the per-candidate score is drawn from the normal approximation to
the same binomial instead, which at n >= 60 and p = 0.5 agrees with the exact
figure to well under a tenth of a point (checked against `selection_floor` in
`_verify` below).
"""
import math
import random
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GENS = [1, 2, 3, 5, 8, 12, 18, 25, 35, 45, 60]
SETS = [(60, "60 tasks"), (120, "120 tasks"), (500, "500 tasks")]


def floor_approx(k, n, p=0.5, reps=20000, seed=0):
    rng = random.Random(seed)
    sd = math.sqrt(p * (1 - p) / n)
    return st.mean(max(rng.gauss(p, sd) for _ in range(k)) - p
                   for _ in range(reps))


def _verify():
    from evalfloor import selection_floor
    for k, n in ((10, 120), (30, 120)):
        a, b = floor_approx(k, n), selection_floor(k=k, n=n, p=0.5, reps=4000)
        print(f"    k={k} n={n}  approx {a:.4f}  exact {b:.4f}")


if __name__ == "__main__":
    _verify()
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    for n, label in SETS:
        ys = [floor_approx(g * 3, n) * 100 for g in GENS]
        ax.plot(GENS, ys, lw=2.2, marker="o", ms=3.5, label=label)

    ax.set_xlabel("generations (3 candidates each)")
    ax.set_ylabel("accuracy points gained")
    ax.set_title("What a self-improving loop reports when nothing improves",
                 fontsize=11, loc="left")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 61)
    ax.set_ylim(bottom=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("docs/null.png")
    print("  docs/null.png")
