"""The same loop, on an agent that cannot possibly improve.

This is the example worth reading twice. The backend here ignores the policy
entirely and answers at random, so no mutation can help and the true gain of
every candidate is exactly zero. The loop still reports a gain, because the
maximum of many noisy scores is above the mean of the thing being measured,
and the more generations you run the larger that gain gets.

Any self-improving agent loop that keeps the best-scoring candidate has this
property. Most of them do not report it.
"""
import random

from jev_evolve import Policy, choice, evolve
from jev_evolve.demo import INTENTS, tickets
from jev_evolve.mutate import mutate_criteria_from_examples, mutate_threshold


class Coinflip:
    """Answers uniformly at random. Reads neither the state nor the criteria."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def decide(self, state, questions):
        out = {}
        for name, q in questions.items():
            opts = list((q.get("criteria") or {}).keys())
            p = [self.rng.random() for _ in opts]
            tot = sum(p)
            probs = {k: v / tot for k, v in zip(opts, p)}
            out[name] = {"probabilities": probs,
                         "choice": max(probs, key=probs.get)}
        return out


train, held = tickets(120, seed=1), tickets(120, seed=2)
policy = Policy({"intent": choice("Pick the intent.",
                                  {k: k.replace("_", " ") for k in INTENTS})})

for gens in (5, 20, 60):
    res = evolve(policy, Coinflip(seed=11), train, score=lambda e: float(
        e.decisions[0].choice == e.label),
        operators=[mutate_criteria_from_examples(INTENTS), mutate_threshold()],
        generations=gens, candidates=3, heldout=held, seed=5, verbose=False)
    print(f"  {gens:2d} generations  apparent gain {res.gain:+.3f}"
          f"   floor {res.floor.floor:+.3f}   verdict {res.verdict}")

print("""
  The true gain is zero at every row: the backend never reads the policy.
  The apparent gain grows with the number of generations, the floor grows
  with it, and the held-out test is what refuses to confirm any of it.""")
