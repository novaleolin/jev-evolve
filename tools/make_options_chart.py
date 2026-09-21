"""Draw docs/options.png from experiments/option_count_results.json."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open("experiments/option_count_results.json"))
xs = [r["options"] for r in rows]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=200)

a1.plot(xs, [r["excess_flip_rate"] * 100 for r in rows], lw=2.2, marker="o",
        ms=5, color="#c2452d")
a1.set_xlabel("options in the choice")
a1.set_ylabel("answers decided by option order (%)")
a1.set_title("Order dependence grows with the option set",
             fontsize=10.5, loc="left")
a1.set_ylim(0, 105)

a2.plot(xs, [r["as_written"] for r in rows], lw=2.2, marker="o", ms=5,
        label="as written", color="#8a8a8a")
a2.plot(xs, [r["marginalized"] for r in rows], lw=2.2, marker="o", ms=5,
        label="ordering marginalized", color="#1f6f8b")
a2.set_xlabel("options in the choice")
a2.set_ylabel("accuracy")
a2.set_title("Each row is a different task: compare the pair",
             fontsize=10.5, loc="left")
a2.legend(frameon=False, fontsize=9)

for ax in (a1, a2):
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xticks(xs)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("docs/options.png")
print("docs/options.png")
