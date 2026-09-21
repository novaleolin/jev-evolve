"""Where a typed decision actually gets made.

Three backends ship. `JevBackend` calls a hosted typed-decision endpoint.
`LocalBackend` reads option logits from any causal LM in one prefill, so an
evolution loop of a few thousand decisions runs offline at no marginal cost.
`RuleBackend` answers from a Python function and needs nothing installed,
which is what the examples and the test suite run on.

Keeping the search loop on a local or rule backend is not only thrift.
Hosted providers generally forbid using output to train or build a competing
model, so evolving a schema against something you own and merely *checking*
the winner against the hosted endpoint stays on the right side of that line.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol


class Backend(Protocol):
    def decide(self, state: dict, questions: dict) -> dict[str, Any]:
        """name -> {"probabilities": {...}, "choice": str} or {"noul": float}."""
        ...


def render_state(state: dict) -> str:
    lines = []
    for k, v in state.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def timed(backend: Backend, state: dict, questions: dict):
    """Run a backend and return (answers, seconds).

    Latency is recorded per decision rather than per episode because the
    reason to use a typed decision model at all is usually the tail, and an
    episode total hides which decision owns it.
    """
    t0 = time.time()
    ans = backend.decide(state, questions)
    return ans, time.time() - t0


class RuleBackend:
    """Answer from a callable. No model, no key, no network.

    The callable receives `(state, name, question)` and returns either an
    option name, a float in [0, 1] for a noul, or a dict of option ->
    probability. Anything it returns that is not a valid option is reported
    as an abstention rather than silently coerced, so a broken rule shows up
    as a broken rule instead of as a wrong answer.
    """

    def __init__(self, fn: Callable[[dict, str, dict], Any], noise: float = 0.0,
                 seed: int = 0):
        self.fn = fn
        self.noise = noise
        self._rng = __import__("random").Random(seed)

    def decide(self, state: dict, questions: dict) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, q in questions.items():
            raw = self.fn(state, name, q)
            if q.get("type") == "noul":
                v = float(raw) if raw is not None else 0.5
                if self.noise:
                    v = min(1.0, max(0.0, v + self._rng.gauss(0, self.noise)))
                out[name] = {"noul": v}
                continue
            opts = list((q.get("criteria") or {}).keys())
            if isinstance(raw, dict):
                probs = {k: float(raw.get(k, 0.0)) for k in opts}
            else:
                probs = {k: (0.9 if k == raw else 0.1 / max(1, len(opts) - 1))
                         for k in opts}
            if self.noise:
                probs = {k: max(1e-9, v + self._rng.gauss(0, self.noise))
                         for k, v in probs.items()}
            tot = sum(probs.values()) or 1.0
            probs = {k: v / tot for k, v in probs.items()}
            out[name] = {"probabilities": probs,
                         "choice": max(probs, key=probs.get) if probs else None}
        return out


class LocalBackend:
    """Score options by next-token logits from one prefill.

    Options are presented as lettered lines and the logits of the marker
    tokens are softmaxed against each other, so the model never generates and
    the answer is a probability vector over exactly the allowed options.
    Nouls do the same over yes/no.

    This is not trying to match a purpose-trained decision model's accuracy.
    It is trying to give the SEARCH a cheap, deterministic, offline signal.
    Whether a schema found this way survives on the hosted model is an
    empirical question, which is why `Evolution.report()` keeps the two
    numbers separate instead of assuming they agree.
    """

    @staticmethod
    def _labels(n: int) -> list[str]:
        """Option markers. Letters read better; numbers scale past 26."""
        if n <= 26:
            return [chr(65 + i) for i in range(n)]
        return [str(i + 1) for i in range(n)]

    def __init__(self, model="Qwen/Qwen2.5-0.5B-Instruct", device=None,
                 max_len=3072):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float32 if self.device == "cpu" else torch.float16
        ).to(self.device).eval()
        self.max_len = max_len
        self._cache: dict[Any, Any] = {}

    def _score(self, prompt: str, surfaces: list[str]) -> list[float]:
        """Probability over exactly the allowed options, from one prefill.

        Only the resulting probabilities are cached, never the logits row: a
        full-vocabulary tensor costs about 600 KB per distinct prompt and on
        an accelerator that exhausts an 18 GB device part-way through a real
        run. The answer for a (prompt, options) pair is all anything
        downstream ever needs.
        """
        key = (prompt, tuple(surfaces))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        ids = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=self.max_len).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**ids).logits[0, -1]
        vals = []
        for s in surfaces:
            tid = self.tok.encode(s, add_special_tokens=False)
            vals.append(float(logits[tid[0]]) if tid else -1e9)
        del logits
        out = self.torch.softmax(self.torch.tensor(vals), dim=0).tolist()
        if len(self._cache) < 200000:
            self._cache[key] = out
        return out

    def decide(self, state: dict, questions: dict) -> dict[str, Any]:
        head = render_state(state)
        out: dict[str, Any] = {}
        for name, q in questions.items():
            if q.get("type") == "choice":
                opts = list((q.get("criteria") or {}).items())
                if not opts:
                    out[name] = {"probabilities": {}, "choice": None}
                    continue
                marks = self._labels(len(opts))
                body = "\n".join(f"{marks[i]}) {k}: {v}"
                                 for i, (k, v) in enumerate(opts))
                p = self._score(
                    f"{head}\n\n{q.get('instructions','')}\n{body}\n\nAnswer:",
                    [f" {m}" for m in marks])
                probs = {k: p[i] for i, (k, _v) in enumerate(opts)}
                out[name] = {"probabilities": probs,
                             "choice": max(probs, key=probs.get)}
            else:
                p = self._score(
                    f"{head}\n\n{q.get('instructions','')}\n\nAnswer (yes or no):",
                    [" yes", " no"])
                out[name] = {"noul": p[0]}
        return out


class JevBackend:
    """A hosted typed-decision endpoint.

    The default points at the OpenRouter decisions API, which takes
    `{model, state, questions}` and returns `answers`. Any provider with the
    same shape works by passing `endpoint`; nothing else in this package
    knows which vendor is behind it.

    Failures are raised, not swallowed. A backend that returns an empty dict
    on a 429 turns a rate limit into an accuracy drop, and an evolution loop
    will then happily select the schema that was rate-limited least.
    """

    ENDPOINT = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, model="typesafe/jev-1.13", api_key=None, timeout=90,
                 endpoint=None, retries=2):
        self.model = model
        self.timeout = timeout
        self.endpoint = endpoint or self.ENDPOINT
        self.retries = retries
        self.key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.key:
            raise SystemExit(
                "JevBackend needs an API key: set OPENROUTER_API_KEY, or pass "
                "api_key=. To run without one, use RuleBackend or LocalBackend.")
        self.calls = 0

    def decide(self, state: dict, questions: dict) -> dict[str, Any]:
        body = json.dumps({"model": self.model, "state": state,
                           "questions": questions}).encode()
        last = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                self.endpoint, data=body,
                headers={"Authorization": "Bearer " + self.key,
                         "Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    self.calls += 1
                    return json.loads(r.read().decode()).get("answers", {})
            except urllib.error.HTTPError as e:
                last = e
                if e.code not in (429, 500, 502, 503, 504):
                    raise
            except urllib.error.URLError as e:
                last = e
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"decision endpoint failed after "
                           f"{self.retries + 1} attempts: {last}")
