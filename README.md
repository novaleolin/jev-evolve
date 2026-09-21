<div align="center">

# Jev-Evolve: self-improving agents that know what's noise

**Typed decisions instead of generated text. A policy that evolves from the agent's own mistakes. And a number telling you how much of the improvement was luck.**

[![PyPI](https://img.shields.io/pypi/v/jev-evolve?logo=pypi&logoColor=white)](https://pypi.org/project/jev-evolve/)
[![Python](https://img.shields.io/pypi/pyversions/jev-evolve?logo=python&logoColor=white)](https://pypi.org/project/jev-evolve/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/novaleolin/jev-evolve/tests.yml?branch=main&label=tests&logo=github)](https://github.com/novaleolin/jev-evolve/actions)

**[Quickstart](#quickstart) · [The null loop](#the-part-everyone-skips) · [How it works](#how-it-works) · [API](#api) · [FAQ](#faq) · [简体中文](README.zh-CN.md)**

</div>

---

## What is Jev-Evolve?

An agent framework where every branch the agent takes is a typed question:
pick one of these options, yes or no, score this. Not generated text that
something downstream parses.

That has a consequence worth the whole library. The agent's policy becomes a
data structure, so a loop can mutate it, and every decision comes with
calibrated probabilities, so the loop can see exactly which decision lost
which episode. The agent improves itself from its own trace.

And because a loop that keeps the best of many candidates reports a gain even
when nothing improved, every run prints the gain a loop of the same size
would have produced on an agent that never got better.

| | |
|---|---|
| **Typed decisions** | `choice`, `noul`, `score`. Works with hosted decision endpoints, a local model, or a plain Python function |
| **Evolves from its trace** | mutations aim at the option pair the agent actually confused, not at random |
| **Reports its own noise** | a null loop in `examples/null_loop.py` reports +0.058 after 60 generations, with a true gain of zero |
| **Runs offline** | the quickstart needs no API key, no model download, no network |

## Quickstart

```bash
pip install jev-evolve
python examples/quickstart.py     # 20 seconds, no API key, no downloads
```

A ticket triage agent with one decision point, described the way everyone
describes it first: each option labelled with its own name.

```python
from jev_evolve import Policy, choice, evolve, run_policy
from jev_evolve.demo import INTENTS, OverlapBackend, tickets
from jev_evolve.mutate import mutate_criteria_from_errors, mutate_threshold

train, held = tickets(120, seed=1), tickets(120, seed=2)
policy = Policy({"intent": choice(
    "Which intent does this support ticket belong to?",
    {k: k.replace("_", " ") for k in INTENTS})})

backend = OverlapBackend(seed=7)
score = lambda ep: float(ep.decisions[0].choice == ep.label)

# One pass first, so the mutations have real mistakes to aim at.
trace = run_policy(policy, backend, train, score)

res = evolve(policy, backend, train, score,
             operators=[mutate_criteria_from_errors(trace), mutate_threshold()],
             generations=12, candidates=3, heldout=held, seed=3)
print(res.report())
```

```
  generations            13
  candidates scored      37
  baseline (train)       0.475
  winner   (train)       0.742   apparent gain +0.267
  selection floor        +0.097   <- the gain is above the floor
  baseline (held-out)    0.467
  winner   (held-out)    0.633   real gain +0.167
  held-out paired        30 fixed / 10 broken   p=0.0022

  verdict: CREDIBLE
```

Thirty-seven candidates were scored, not the four that were kept. The floor
is computed from all thirty-seven, because that is how many chances the loop
had to get lucky.

## The part everyone skips

Run the same loop on a backend that answers at random and ignores the policy
entirely. No mutation can help it. The true gain is zero.

```bash
python examples/null_loop.py
```

```
   5 generations  apparent gain +0.017   floor +0.064   verdict NOT CONFIRMED
  20 generations  apparent gain +0.050   floor +0.086   verdict NOT CONFIRMED
  60 generations  apparent gain +0.058   floor +0.101   verdict NOT CONFIRMED
```

![what a loop reports when nothing improves](docs/null.png)

The reported gain grows with the number of generations, on an agent that is
not improving at all. This is not a bug in this loop. It is what happens
whenever you keep the best of several noisy measurements, and it applies to
every self-improving agent that selects on an eval score.

Jev-Evolve handles it in two ways. It prints the floor, so you can see the size
of the effect for your own setup. And it decides the verdict on a held-out
split the search never touched, using a paired sign test, so a run is
`CREDIBLE` only on evidence that selection could not have manufactured.

The arithmetic comes from [evalfloor](https://github.com/novaleolin/evalfloor),
which is a dependency rather than a copy, so there is one implementation
instead of two that drift.

## How it works

Four pieces, each independently usable.

**Policy.** The decision points, as data. Instruction text, per-option
criteria, which state fields each point can see, and the confidence
threshold below which it abstains. All four are searchable. Three of them
are invisible to a prompt optimiser.

```python
from jev_evolve import Policy, choice, noul

policy = Policy({
    "in_scope": noul("Is this about a bank account?", threshold=0.6),
    "intent":   choice("Which intent?", {"lost_card": "...", "top_up": "..."},
                       threshold=0.3, state_fields=["text", "channel"]),
}, order=["in_scope", "intent"])
```

**Agent.** A loop over those points. `act(state, answer)` applies each answer
and is where your tools live. Return `{jev_evolve.STOP: True}` to finish early.

```python
from jev_evolve import Agent, RuleBackend

def act(state, answer):
    if answer.name == "in_scope" and not answer:
        return {jev_evolve.STOP: True, "outcome": "handoff"}
    return {answer.name: answer.choice}

episode = Agent(policy, backend, act=act).run("t1", {"text": "lost my card"})
```

`answer` is falsy when the point abstained or answered no, so `if not answer`
covers both. `answer.margin` is the gap to the runner-up, which is the field
worth sorting your errors by.

**Trace.** Every decision, its probabilities and its latency, saved as JSONL.
Three readers come with it:

```python
jev_evolve.confusions(trace)       # (point, picked, should have been) -> count
jev_evolve.point_accuracy(trace)   # which decision point owns the loss
jev_evolve.overconfident(trace)    # wrong and sure, the ones thresholds can't catch
jev_evolve.cost(trace)             # decisions and seconds per episode
```

**Evolve.** Generations of mutants, scored, with the best kept.

```python
from jev_evolve.mutate import default_operators

ops = default_operators(available_fields=["text", "channel", "tier"],
                        examples_by_label=INTENTS, trace=trace)
res = evolve(policy, backend, train, score, ops, heldout=held)
res.best.save("policy.json")
```

## Mutations

Five edit the policy blind. Two read the agent's own trace, and those are
the reason the trace exists.

| operator | what it changes | needs |
| :--- | :--- | :--- |
| `mutate_threshold` | when a point abstains instead of guessing | nothing |
| `mutate_order` | which point runs first, and so what later points see | nothing |
| `mutate_state_fields` | what one point is allowed to see | field names |
| `mutate_criteria_from_examples` | describes an option by real inputs | labelled data |
| `mutate_criteria_from_errors` | describes an option by inputs it **missed** | a trace |
| `mutate_from_confusions` | rewrites the option most often picked by mistake | a trace, a rewriter |
| `mutate_instructions` | rewrites a point's instruction text | a rewriter |

A rewriter is any `(text, slot, rng) -> text`. An LLM is the obvious one, and
a template or a hand-written list works too. The `rng` is part of the
contract: a rewriter that reaches for the global `random` makes the run
unreproducible, and a tool whose job is telling you how much of a gain is
real cannot give a different answer each time you ask.

The first four need no LLM at all, so a generation costs nothing and there is
never a budget reason to skip the check.

## Backends

```python
from jev_evolve import RuleBackend, JevBackend
```

| backend | what it is | cost |
| :--- | :--- | :--- |
| `RuleBackend` | a Python function. Tests, baselines, hybrid policies | free |
| `OverlapBackend` | `jev_evolve.demo`, answers by word overlap. Examples and CI | free |
| `LocalBackend` | option logits from any causal LM, one prefill, no generation | local GPU |
| `JevBackend` | a hosted typed-decision endpoint | per call |

`LocalBackend` needs `pip install "jev-evolve[local]"`. Everything else in the
package works with no extra dependency, and `import jev_evolve` pulls in no
model library.

`JevBackend` defaults to the OpenRouter decisions API and takes `endpoint=`
for anything with the same `{model, state, questions}` shape. Nothing else in
the package knows which vendor is behind it.

A note on where to run the loop. Hosted providers generally forbid using
output to train or build a competing model. Evolving a policy against a local
or rule backend and merely checking the winner against the hosted endpoint
keeps you on the right side of that, and it is also much cheaper.

## API

```python
from jev_evolve import Policy, Point, choice, noul       # the policy
from jev_evolve import Agent, Answer, STOP               # the loop
from jev_evolve import Trace, Episode, Decision          # the record
from jev_evolve import evolve, run_policy, Result        # the search
from jev_evolve import confusions, point_accuracy, overconfident, cost

evolve(policy, backend, tasks, score, operators,
       generations=20, candidates=4, heldout=None, act=None, seed=0)
# -> Result: .best .gain .floor .heldout .credible .verdict .report()

run_policy(policy, backend, tasks, score) -> Trace
Agent(policy, backend, act=None, route=None).run(task_id, state, label) -> Episode
```

A task is `(id, state, label)` or `{"id":..., "state":..., "label":...}`.
`score(episode) -> float` is yours; 0/1 keeps the floor arithmetic exact.

## Limits

The selection floor assumes candidates are evaluated independently on a 0/1
metric. Candidates in an evolutionary loop are correlated by descent, which
makes the true floor higher than reported, so the printed number is a lower
bound. A gain under it is under the true floor as well. A gain over it still
needs the held-out test.

Attribution needs to know which decision point should have answered what.
With one choice point in the policy, `default_gold` handles it. With more,
pass your own `gold(episode) -> {point: answer}`. It refuses to guess, because
a guessed attribution aims every later mutation at the wrong decision.

Latency numbers come from wall-clock time around the backend call and include
network time.

## FAQ

**How is this different from DSPy or a prompt optimiser?**
Those search over prompt text and few-shot examples. Here the unit is a typed
decision, so the search also covers option criteria, per-point state
visibility, confidence thresholds and decision order. And the run ends with a
verdict rather than a best score.

**Do I need a Jev or System One model?**
No. `RuleBackend` and `OverlapBackend` need nothing, `LocalBackend` runs any
causal LM by reading option logits. A hosted typed-decision endpoint is one
backend of four.

**My agent already works. What do I get?**
`confusions()` and `point_accuracy()` on one recorded run, which usually shows
that a single decision point owns most of the loss. That is worth the
afternoon on its own, before any evolution.

**Why is my run UNDERPOWERED?**
An exact sign test over `d` disagreements cannot return a p-value below
2^(1-d), so five or fewer can never reach 0.05. The winner may well be
better; your held-out split is too small to show it. Use more held-out data.

**Can I evolve against the hosted endpoint directly?**
You can, and it will be slow and expensive, and the provider's terms probably
do not want you to. Evolve locally, validate hosted.

**Does the floor mean my improvement is fake?**
It means a search of that size gets that much for free. The verdict comes
from the held-out sign test, not the floor. A gain under the train floor can
still be credible, and `Result.credible` will say so.

MIT.
