"""The generator's own text encoder, used as a typed-decision judge.

Qwen-Image-2.1 ships its text encoder as a complete Qwen3-VL: language
model, vision tower, lm_head and processor. So the weights that condition
the generator can also grade its output, with no second download. Answers
are read from next-token logits as yes/no probabilities, which makes this a
Backend in the jev-evolve sense: `permutation_sensitivity`, thresholds and
`Marginalized` all apply to it unchanged.

Stated up front: a judge that shares weights with the generator's
conditioning is not independent of it. It is the cheapest first judge, and
the loop keeps a slot for a second, unrelated one.
"""
from __future__ import annotations

import os
from typing import Any


class VLMJudge:
    """Yes/no questions about an image, answered as calibrated-ish logits.

    `decide(state, questions)` follows the jev-evolve Backend protocol. The
    state carries `image` (a PIL image or a path); each question is a `noul`
    whose `instructions` is the question text. Returned `noul` is P(yes).
    """

    YES = (" yes", "Yes", " Yes", "yes")
    NO = (" no", "No", " No", "no")

    def __init__(self, model_dir: str, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(os.path.join(model_dir, "processor"))
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(model_dir, "text_encoder"), dtype=torch.bfloat16
        ).to(device).eval()
        self.device = device
        tok = self.proc.tokenizer
        self.yes_ids = sorted({tok.encode(s, add_special_tokens=False)[0] for s in self.YES})
        self.no_ids = sorted({tok.encode(s, add_special_tokens=False)[0] for s in self.NO})
        self.calls = 0

    def _image(self, state):
        from PIL import Image
        img = state["image"]
        return Image.open(img).convert("RGB") if isinstance(img, str) else img

    def p_yes(self, image, question: str) -> float:
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question + " Answer yes or no."}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False)
        inputs = self.proc(text=[text], images=[image],
                           return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.vlm(**inputs).logits[0, -1].float()
        two = self.torch.stack([logits[self.yes_ids].logsumexp(0),
                                logits[self.no_ids].logsumexp(0)])
        self.calls += 1
        return float(self.torch.softmax(two, 0)[0])

    def decide(self, state: dict, questions: dict) -> dict[str, Any]:
        img = self._image(state)
        return {name: {"noul": self.p_yes(img, q.get("instructions", name))}
                for name, q in questions.items()}
