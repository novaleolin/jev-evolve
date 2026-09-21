"""Is the schema measuring its criteria, or the option order?

15 seconds. No API key, no model download, no network.

The backend here reads the criteria text and also carries a mild preference
for whatever is listed first, which is what real option-scoring models do.
The check separates the two, and the wrapper removes the second.
"""
from jev_evolve import (Marginalized, Policy, choice, permutation_sensitivity,
                        run_policy)
from jev_evolve.demo import INTENTS, OverlapBackend, tickets


class PositionBiased:
    """A real scorer with a thumb on the first option."""

    def __init__(self, strength=0.35, seed=0):
        self.inner = OverlapBackend(seed=seed, noise=0.05)
        self.strength = strength

    def decide(self, state, questions):
        out = self.inner.decide(state, questions)
        for name, a in out.items():
            probs = a.get("probabilities")
            if not probs:
                continue
            # a decaying prior over position, blended with the real scores
            n = len(probs)
            prior = {k: (n - i) / (n * (n + 1) / 2)
                     for i, k in enumerate(probs)}
            mixed = {k: (1 - self.strength) * probs[k]
                        + self.strength * prior[k] for k in probs}
            tot = sum(mixed.values()) or 1.0
            mixed = {k: v / tot for k, v in mixed.items()}
            out[name] = {"probabilities": mixed,
                         "choice": max(mixed, key=mixed.get)}
        return out


tasks = tickets(120, seed=2)
score = lambda ep: float(ep.decisions[0].choice == ep.label)

# Two schemas. The first is what everyone writes before measuring anything:
# each option described by its own name, which says nothing the label did not.
# The second describes each option with real inputs that belong to it.
WEAK = {k: k.replace("_", " ") for k in INTENTS}
STRONG = {k: "Examples: " + " | ".join(f'"{x}"' for x in v[:2])
          for k, v in INTENTS.items()}

for name, crit, strength in (("thin criteria, strong position prior", WEAK, 0.8),
                             ("real criteria, mild position prior", STRONG, 0.35)):
    policy = Policy({"intent": choice(
        "Which intent does this support ticket belong to?", dict(crit))})
    backend = PositionBiased(strength=strength)
    print(f"\n{'=' * 66}\n  {name}\n{'=' * 66}\n")

    s = permutation_sensitivity(policy, backend, tasks, score, k=8)
    print(s.report())

    plain = run_policy(policy, backend, tasks, score).score
    fixed = run_policy(policy, Marginalized(backend, k=8), tasks, score).score
    print(f"\n    as written              {plain:.3f}")
    print(f"    ordering marginalized   {fixed:.3f}   {fixed - plain:+.3f}")

print("""
  The check discriminates. It is not a warning that always fires: a schema
  whose criteria carry the decision passes it, and marginalizing such a
  schema buys nothing, which is the correct answer and not a failure.""")
