"""A prompt-evolution loop for a text-to-image model, with the noise counted.

The cheap part of any such loop is proposing prompt rewrites. The expensive
part is scoring them: every candidate costs a generation per prompt per seed,
so eval sets end up small and the loop ends up selecting the maximum of a
few dozen noisy numbers. That is exactly the regime where a 30-candidate
search on 30 prompts reports a double-digit gain on nothing.

So this loop does three things a typical one does not: it averages over
seeds, so a rewrite is not credited for a lucky sample; it reports the
selection floor for the number of candidates actually scored; and it decides
on held-out prompts the search never saw. Everything else is plumbing.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from geneval import Item


class Generator(Protocol):
    def generate(self, prompt: str, seed: int) -> "PIL.Image.Image": ...


class QwenImageGenerator:
    """Qwen-Image-2.1 through diffusers, with an on-disk image cache.

    The cache key is (prompt, seed, size, steps). A prompt-evolution loop
    re-scores the incumbent every generation and re-visits rewrites it has
    seen, and at tens of seconds per image, not caching is the difference
    between a loop that runs and one that does not.
    """

    def __init__(self, model_dir: str, cache_dir: str, size=(1024, 1024),
                 steps: int = 40, device: str = "cuda"):
        import torch
        from diffusers import QwenImage21Pipeline
        self.torch = torch
        self.pipe = QwenImage21Pipeline.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16).to(device)
        self.size, self.steps, self.device = size, steps, device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.generated = 0
        self.seconds = 0.0

    def _key(self, prompt, seed):
        blob = json.dumps([prompt, seed, self.size, self.steps])
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def generate(self, prompt: str, seed: int):
        from PIL import Image
        path = os.path.join(self.cache_dir, self._key(prompt, seed) + ".png")
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
        t0 = time.time()
        g = self.torch.Generator(self.device).manual_seed(seed)
        img = self.pipe(prompt=prompt, width=self.size[0], height=self.size[1],
                        num_inference_steps=self.steps, generator=g).images[0]
        self.seconds += time.time() - t0
        self.generated += 1
        img.save(path)
        return img


@dataclass
class Scored:
    """One prompt variant's score on one item, averaged over seeds."""
    item_id: str
    prompt: str
    per_seed: list = field(default_factory=list)
    answers: list = field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(self.per_seed) / len(self.per_seed) if self.per_seed else 0.0


def score_variant(gen: Generator, judge, item: Item, prompt: str,
                  seeds: Sequence[int], threshold: float = 0.5) -> Scored:
    """Generate once per seed, ask every question, apply GenEval's all-or-
    nothing rule per image, average across seeds.

    The judge returns P(yes) per question. Thresholding at 0.5 is the
    obvious default and it is one of the things the invariance checks in
    this repo are for: a judge whose yes/no flips under rephrasing is not
    scoring the image.
    """
    out = Scored(item.id, prompt)
    qs = {q.text: {"type": "noul", "instructions": q.text} for q in item.questions}
    for seed in seeds:
        img = gen.generate(prompt, seed)
        raw = judge.decide({"image": img}, qs)
        ans = {k: raw[k]["noul"] >= threshold for k in qs}
        out.answers.append({k: raw[k]["noul"] for k in qs})
        out.per_seed.append(item.score(ans))
    return out


Rewriter = Callable[[str, Item, "random.Random"], str]


def evaluate(gen, judge, items: Sequence[Item], rewrite: Rewriter | None,
             seeds: Sequence[int], rng) -> tuple[float, list]:
    """Mean score over items for one rewriter (None = the prompt as written)."""
    rows = []
    for it in items:
        p = rewrite(it.prompt, it, rng) if rewrite else it.prompt
        rows.append(score_variant(gen, judge, it, p, seeds))
    return sum(r.score for r in rows) / len(rows), rows


def hits(rows: list, threshold: float = 0.5) -> list:
    """Per-item 0/1 for the paired test: an item counts as a hit when the
    seed-averaged score clears the threshold. Coarse on purpose, so the sign
    test's arithmetic stays exact."""
    return [int(r.score >= threshold) for r in rows]


@dataclass
class LoopResult:
    base: float
    best: float
    best_name: str
    scores: dict
    floor: object = None
    heldout: object = None
    base_heldout: float | None = None
    best_heldout: float | None = None
    images: int = 0
    seconds: float = 0.0

    @property
    def gain(self) -> float:
        return self.best - self.base

    def report(self) -> str:
        L = [f"  candidates scored      {len(self.scores)}",
             f"  images generated       {self.images}   ({self.seconds / 60:.0f} min)",
             f"  as written (train)     {self.base:.3f}",
             f"  best rewriter (train)  {self.best:.3f}   {self.best_name}"
             f"   apparent gain {self.gain:+.3f}"]
        if self.floor is not None:
            rel = "BELOW" if self.gain < self.floor.floor else "above"
            L.append(f"  selection floor        {self.floor.floor:+.3f}"
                     f"   <- the gain is {rel} the floor")
        if self.heldout is not None:
            L += [f"  as written (held-out)  {self.base_heldout:.3f}",
                  f"  best (held-out)        {self.best_heldout:.3f}"
                  f"   real gain {self.best_heldout - self.base_heldout:+.3f}",
                  f"  paired                 {self.heldout.wins} fixed /"
                  f" {self.heldout.losses} broken   p={self.heldout.p_value:.4f}"]
            v = ("CREDIBLE" if self.heldout.confirmed else
                 "UNDERPOWERED" if getattr(self.heldout, "underpowered", False)
                 else "NOT CONFIRMED")
            L += ["", f"  verdict: {v}"]
        return "\n".join(L)


def run_loop(gen, judge, train: Sequence[Item], rewriters: dict,
             seeds: Sequence[int], heldout: Sequence[Item] | None = None,
             seed: int = 0) -> LoopResult:
    """Score every rewriter on the training prompts, keep the best, and say
    how much of that a search this size gets for free.

    Every rewriter is scored, including the ones that lose, and all of them
    count toward the floor. `rewriters` should include at least one that
    cannot help (see `null_rewriters`), because a loop that has never been
    run on a null is a loop whose output nobody has calibrated.
    """
    import random
    from evalfloor import check, confirm
    rng = random.Random(seed)
    base, base_rows = evaluate(gen, judge, train, None, seeds, rng)
    scores = {"as written": base}
    for name, rw in rewriters.items():
        scores[name], _ = evaluate(gen, judge, train, rw, seeds, random.Random(seed))
    best_name = max(scores, key=scores.get)
    res = LoopResult(base=base, best=scores[best_name], best_name=best_name,
                     scores=scores)
    res.floor = check(list(scores.values()), n_examples=len(train), baseline=base)
    if heldout and best_name != "as written":
        hb, hb_rows = evaluate(gen, judge, heldout, None, seeds, random.Random(seed))
        hw, hw_rows = evaluate(gen, judge, heldout, rewriters[best_name], seeds,
                               random.Random(seed))
        res.base_heldout, res.best_heldout = hb, hw
        res.heldout = confirm(hits(hb_rows), hits(hw_rows))
    res.images = getattr(gen, "generated", 0)
    res.seconds = getattr(gen, "seconds", 0.0)
    return res


def null_rewriters() -> dict:
    """Rewrites that cannot change what the image should contain.

    Run these first. Whatever gain they report is the size of the noise the
    loop is selecting on, and any real rewriter has to clear it.
    """
    import random
    tails = [", high quality", ", detailed", ", photo", ", 4k", ", sharp focus",
             ", realistic", ", well lit", ", centered composition"]

    def suffix(p, it, rng: random.Random):
        return p + rng.choice(tails)

    def reorder(p, it, rng: random.Random):
        return p.replace("a photo of ", "photo: ", 1)

    return {"null: random suffix": suffix, "null: reworded prefix": reorder}
