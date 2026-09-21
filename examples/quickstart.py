"""A triage agent that rewrites its own decision criteria.

Runs in about 20 seconds. No API key, no model download, no network.

The agent has one decision point: which intent a support ticket belongs to.
It starts the way everyone starts, with each option described by its own
name, which tells the model nothing the label did not already say. It then
evolves that policy against its own mistakes and reports how much of the
improvement survives data the search never ran on.
"""
import jevolve
from jevolve import choice, evolve, Policy
from jevolve.demo import INTENTS, OverlapBackend, tickets
from jevolve.mutate import (mutate_criteria_from_errors,
                            mutate_criteria_from_examples, mutate_state_fields,
                            mutate_threshold)

train = tickets(120, seed=1)
held = tickets(120, seed=2)

# The starting policy: every option described by its own label. This is not a
# straw man, it is what a schema looks like before anyone has measured it.
policy = Policy({"intent": choice(
    "Which intent does this support ticket belong to?",
    {k: k.replace("_", " ") for k in INTENTS})})

backend = OverlapBackend(seed=7)


def score(ep):
    return float(ep.decisions[0].choice == ep.label)


# Two of these operators read the agent's own trace, which is why the trace
# exists. The first generation is run before any of them fire, so the errors
# they draw from are real ones the agent made under this exact policy.
base = jevolve.run_policy(policy, backend, train, score)
print(f"  starting accuracy {base.score:.3f}")
print(f"  worst confusions   {jevolve.confusions(base).most_common(3)}\n")

ops = [
    mutate_criteria_from_errors(base),
    mutate_criteria_from_examples({k: v for k, v in INTENTS.items()}),
    mutate_state_fields(["text", "channel", "tier"]),
    mutate_threshold(),
]

res = evolve(policy, backend, train, score, ops,
             generations=12, candidates=3, heldout=held, seed=3)

print()
print(res.report())
print()
for name, pt in res.best.points.items():
    print(f"  {name}: threshold {pt.threshold}, sees {pt.state_fields}")
    for k, v in pt.question["criteria"].items():
        print(f"    {k}: {v[:90]}")
