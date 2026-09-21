"""Tests written as attacks on the library's own claims.

Each one names a way the package could be quietly wrong and checks that it
is not. The claim under attack is in the test name.
"""
import json
import random

import pytest

import jevolve
from jevolve import Agent, Policy, choice, evolve, noul, run_policy
from jevolve.demo import INTENTS, OverlapBackend, tickets
from jevolve.backends import RuleBackend
from jevolve.mutate import (default_operators, mutate_criteria_from_errors,
                            mutate_order, mutate_state_fields,
                            mutate_threshold)


def simple_policy():
    return Policy({"intent": choice(
        "Pick the intent.", {k: k.replace("_", " ") for k in INTENTS})})


def acc(ep):
    return float(ep.decisions[0].choice == ep.label)


# --------------------------------------------------------------- the loop


def test_a_null_backend_never_gets_a_credible_verdict():
    """The headline claim. A loop over a backend that ignores the policy
    cannot be confirmed, however many generations it runs."""
    class Coin:
        def __init__(self):
            self.rng = random.Random(3)

        def decide(self, state, questions):
            out = {}
            for n, q in questions.items():
                opts = list((q.get("criteria") or {}).keys())
                p = [self.rng.random() for _ in opts]
                t = sum(p)
                pr = {k: v / t for k, v in zip(opts, p)}
                out[n] = {"probabilities": pr, "choice": max(pr, key=pr.get)}
            return out

    res = evolve(simple_policy(), Coin(), tickets(90, seed=1), acc,
                 [mutate_threshold()], generations=25, candidates=3,
                 heldout=tickets(90, seed=2), seed=1, verbose=False)
    assert not res.credible
    assert res.floor.floor > 0


def test_the_floor_counts_losers_too():
    """Counting only the candidates that were kept understates the floor,
    which is the exact error that makes a loop believe itself."""
    res = evolve(simple_policy(), OverlapBackend(seed=2), tickets(60, seed=1),
                 acc, [mutate_threshold()], generations=6, candidates=3,
                 seed=1, verbose=False)
    kept = sum(1 for g in res.generations if g.accepted)
    assert res._k > kept


def test_the_floor_has_no_vote_on_the_verdict():
    """The floor governs the split that was searched. A winner under the
    train floor but confirmed on held-out data is still credible."""
    res = evolve(simple_policy(), OverlapBackend(seed=7), tickets(120, seed=1),
                 acc, default_operators(examples_by_label=INTENTS),
                 generations=10, candidates=3, heldout=tickets(120, seed=2),
                 seed=3, verbose=False)
    if res.heldout is not None and res.heldout.confirmed:
        assert res.credible
        assert res.verdict == "CREDIBLE"


def test_underpowered_is_not_reported_as_failure():
    """Five one-sided disagreements cannot reach p<0.05. Calling that a
    failure discards real improvements forever."""
    from jevolve.evolve import _confirm
    c = _confirm([0] * 5 + [1] * 50, [1] * 5 + [1] * 50)
    assert c.underpowered and not c.confirmed


def test_a_run_without_heldout_is_untested_not_credible():
    res = evolve(simple_policy(), OverlapBackend(seed=1), tickets(60, seed=1),
                 acc, [mutate_threshold()], generations=3, candidates=2,
                 seed=1, verbose=False)
    assert res.verdict == "UNTESTED" and not res.credible


def test_evolution_is_reproducible():
    """A tool whose business is telling you how much of a gain is real
    cannot hand you a different answer each time you ask."""
    def run():
        return evolve(simple_policy(), OverlapBackend(seed=4),
                      tickets(60, seed=1), acc,
                      default_operators(examples_by_label=INTENTS),
                      generations=6, candidates=2, seed=9, verbose=False)
    a, b = run(), run()
    assert a.best_score == b.best_score
    assert a.best.fingerprint() == b.best.fingerprint()


# --------------------------------------------------------------- the agent


def test_a_point_below_its_threshold_abstains_rather_than_guessing():
    """The whole reason to have calibrated probabilities. An agent that
    always commits has thrown away its one signal for falling back."""
    pol = Policy({"intent": choice("Pick.", {"a": "a", "b": "b"},
                                   threshold=0.99)})
    ep = Agent(pol, RuleBackend(lambda s, n, q: "a")).run("t", {"text": "x"})
    assert ep.decisions[0].choice is None


def test_state_fields_actually_hide_fields_from_the_backend():
    """A restriction that does not restrict makes the search's largest
    operator a no-op while still reporting gains."""
    seen = {}

    def fn(state, name, q):
        seen.update(state)
        return "a"

    pol = Policy({"p": choice("Pick.", {"a": "a"}, state_fields=["text"])})
    Agent(pol, RuleBackend(fn)).run("t", {"text": "x", "secret": "y"})
    assert "secret" not in seen


def test_order_changes_what_later_points_see():
    order = []
    pol = Policy({"a": noul("first?"), "b": noul("second?")}, order=["b", "a"])
    Agent(pol, RuleBackend(lambda s, n, q: order.append(n) or 0.9)).run(
        "t", {"text": "x"})
    assert order == ["b", "a"]


def test_act_can_stop_an_episode_early():
    pol = Policy({"a": noul("q?"), "b": noul("q?"), "c": noul("q?")})
    ep = Agent(pol, RuleBackend(lambda s, n, q: 0.9),
               act=lambda s, a: {jevolve.STOP: True}).run("t", {"text": "x"})
    assert ep.n_decisions == 1


def test_margin_is_top_minus_runner_up_not_top_alone():
    """A 0.51/0.49 decision and a 0.51/0.02/0.02... decision are not equally
    confident, and a loop that treats them alike mis-sorts its own errors."""
    from jevolve.agent import Answer
    assert Answer("x", probabilities={"a": 0.51, "b": 0.49}).margin == \
        pytest.approx(0.02)
    assert Answer("x", noul=0.5).margin == 0.0


# --------------------------------------------------------------- the trace


def test_confusions_point_at_the_pair_that_actually_cost():
    tr = run_policy(simple_policy(), OverlapBackend(seed=7), tickets(120, 1),
                    acc)
    c = jevolve.confusions(tr)
    assert c, "a policy at chance level must produce confusions"
    (point, pred, actual), n = c.most_common(1)[0]
    assert point == "intent" and pred != actual and n > 0


def test_default_gold_refuses_to_guess_attribution():
    """With two choice points the attribution is a guess, and a guessed
    attribution aims every later mutation at the wrong decision."""
    from jevolve.analyze import default_gold
    from jevolve.trace import Decision, Episode
    ep = Episode("t", [Decision(0, "a", "x", {"x": 1.0}),
                       Decision(1, "b", "y", {"y": 1.0})], label="x")
    assert default_gold(ep) == {}


def test_error_driven_mutation_only_touches_options_that_failed():
    tr = run_policy(simple_policy(), OverlapBackend(seed=7), tickets(120, 1),
                    acc)
    op = mutate_criteria_from_errors(tr)
    before = simple_policy()
    after = op(before, random.Random(0))
    changed = [k for k in INTENTS
               if after.points["intent"].question["criteria"][k]
               != before.points["intent"].question["criteria"][k]]
    assert len(changed) == 1


def test_cost_is_reported_so_accuracy_cannot_be_bought_with_latency():
    tr = run_policy(simple_policy(), OverlapBackend(seed=1), tickets(30, 1),
                    acc)
    c = jevolve.cost(tr)
    assert c["episodes"] == 30 and c["decisions_per_episode"] == 1.0


# --------------------------------------------------------------- plumbing


def test_a_policy_survives_a_round_trip_through_json(tmp_path):
    p = simple_policy()
    p.points["intent"].threshold = 0.3
    p.points["intent"].state_fields = ["text"]
    f = tmp_path / "p.json"
    p.save(str(f))
    q = Policy.load(str(f))
    assert q.fingerprint() == p.fingerprint()


def test_a_trace_survives_a_round_trip_through_jsonl(tmp_path):
    tr = run_policy(simple_policy(), OverlapBackend(seed=1), tickets(20, 1),
                    acc)
    f = tmp_path / "t.jsonl"
    tr.save(str(f))
    back = jevolve.Trace.load(str(f))
    assert len(back) == 20
    assert back.score == pytest.approx(tr.score)


def test_importing_jevolve_pulls_in_no_heavy_dependency():
    """`import jevolve` must work with nothing installed but evalfloor."""
    import subprocess
    import sys
    code = ("import sys, jevolve; "
            "assert not {'torch','transformers','numpy'} & set(sys.modules)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_a_mutation_that_cannot_apply_returns_none_rather_than_a_copy():
    """An operator that returns an unchanged policy wastes a candidate slot
    and inflates the floor with a duplicate score."""
    empty = Policy({"p": choice("q", {"a": "a"})})
    assert mutate_order()(empty, random.Random(0)) is None
    assert mutate_state_fields(["text"])(empty, random.Random(0)) is None
