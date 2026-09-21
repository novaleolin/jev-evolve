"""Draw docs/null.png: what a loop reports when nothing is improving.

Simulated rather than run through the agent, because the point is the shape
of the curve across many generation counts and the arithmetic is the same
one `evalfloor.selection_floor` uses.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evalfloor import selection_floor

GENS = [1, 2, 3, 5, 8, 12, 18, 25, 35, 45, 60]
SETS = [(60, "60 tasks"), (120, "120 tasks"), (500, "500 tasks")]

fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
for n, label in SETS:
    ys = [selection_floor(k=g * 3, n=n, p=0.5, reps=4000, seed=1) * 100 for g in GENS]
    ax.plot(GENS, ys, lw=2.2, marker="o", ms=3.5, label=label)

ax.set_xlabel("generations (3 candidates each)")
ax.set_ylabel("accuracy points gained")
ax.set_title("What a self-improving loop reports when nothing improves",
             fontsize=11, loc="left")
ax.grid(alpha=0.25, lw=0.6)
ax.legend(frameon=False, fontsize=9)
ax.set_xlim(1, 60)
ax.set_ylim(bottom=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("docs/null.png")
print("docs/null.png")
