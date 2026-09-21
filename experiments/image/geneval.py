"""GenEval prompts as typed questions with known answers.

GenEval (Ghosh et al., 2023) writes each prompt from slots: which objects,
how many, what colour, where. That structure is the whole reason to use it
here. The questions a judge must answer are generated from the slots, not
decomposed by an LLM, so the ground truth per question is exact and there is
no second model adding noise between the prompt and the score.

Each question is a yes/no `noul` the judge answers from an image. A prompt
is satisfied only when every question is, which is GenEval's own rule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
ARTICLE = {"a": "a", "an": "an"}


def _a(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def _plural(noun: str) -> str:
    """Enough English for the 80 COCO class names GenEval uses."""
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


@dataclass
class Question:
    text: str
    expect: bool
    kind: str


@dataclass
class Item:
    id: str
    prompt: str
    tag: str
    questions: list = field(default_factory=list)

    def score(self, answers: dict) -> float:
        """1.0 only when every question came out as expected. GenEval's rule,
        and the right one: a picture of two red buses when three blue ones
        were asked for is not two-thirds correct."""
        return float(all(bool(answers.get(q.text)) == q.expect
                         for q in self.questions))


def questions_for(meta: dict) -> list:
    """Turn one GenEval metadata row into yes/no questions with answers.

    `include` gives the objects and their counts, colours and relative
    positions; `exclude` (on counting prompts) gives the count that must not
    appear. A question is asked for each fact, plus one negative control per
    prompt asking about an object that is not in it, so a judge that says
    yes to everything scores zero rather than full marks.
    """
    qs = []
    inc = meta.get("include", [])
    for obj in inc:
        cls, n = obj["class"], obj.get("count", 1)
        qs.append(Question(f"Is there {_a(cls)} in the image?", True, "object"))
        if n > 1:
            qs.append(Question(f"Are there exactly {NUM.get(n, n)} "
                               f"{_plural(cls)} in the image?", True, "count"))
        if obj.get("color"):
            qs.append(Question(f"Is the {cls} {obj['color']}?", True, "color"))
        if obj.get("position"):
            rel, other = obj["position"]
            qs.append(Question(f"Is the {cls} {rel} the "
                               f"{inc[other]['class']}?", True, "position"))
    for obj in meta.get("exclude", []):
        n = obj.get("count")
        if n:
            qs.append(Question(f"Are there {NUM.get(n, n)} or more "
                               f"{_plural(obj['class'])} in the image?", False,
                               "count"))
    present = {o["class"] for o in inc}
    for absent in ("giraffe", "umbrella", "refrigerator", "kite"):
        if absent not in present:
            qs.append(Question(f"Is there {_a(absent)} in the image?", False,
                               "control"))
            break
    return qs


def load_geneval(path: str, tags: list | None = None, limit: int | None = None):
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            m = json.loads(line)
            if tags and m["tag"] not in tags:
                continue
            items.append(Item(id=f"g{i}", prompt=m["prompt"], tag=m["tag"],
                              questions=questions_for(m)))
    return items[:limit] if limit else items
