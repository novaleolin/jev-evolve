"""A backend and a dataset that need nothing installed.

Every example and every test in this package runs on these, so `pip install
jev_evolve` is enough to see the whole loop work: no API key, no model download,
no network. They are demonstration devices and are not claimed to predict how
any real model behaves.

`OverlapBackend` is a real scorer, not a scripted one. It answers by token
overlap between the state and each option's written criteria, so an option
described only by its own name is genuinely hard to pick and an option
described by real examples is genuinely easier. That is what makes the demo
loop honest: the evolution has something to find, and if the mutations were
useless the run would report nothing.
"""
from __future__ import annotations

import math
import random
import re

_WORD = re.compile(r"[a-z0-9']+")


def _toks(s: str) -> set:
    return set(_WORD.findall(str(s).lower()))


class OverlapBackend:
    """Score options by word overlap with their criteria text."""

    def __init__(self, temperature: float = 3.0, noise: float = 0.15,
                 seed: int = 0):
        self.temperature = temperature
        self.noise = noise
        self._rng = random.Random(seed)
        self.calls = 0

    def decide(self, state: dict, questions: dict) -> dict:
        text = _toks(" ".join(str(v) for v in state.values()))
        out = {}
        for name, q in questions.items():
            self.calls += 1
            if q.get("type") == "noul":
                hint = _toks(q.get("instructions", ""))
                s = len(text & hint) / (len(hint) or 1)
                out[name] = {"noul": 1 / (1 + math.exp(-(s * 6 - 1.5)))}
                continue
            crit = q.get("criteria") or {}
            raw = {}
            for k, v in crit.items():
                words = _toks(k.replace("_", " ")) | _toks(v)
                raw[k] = (len(text & words) / math.sqrt(len(words) or 1)
                          + self._rng.gauss(0, self.noise))
            if not raw:
                out[name] = {"probabilities": {}, "choice": None}
                continue
            mx = max(raw.values())
            ex = {k: math.exp((v - mx) * self.temperature) for k, v in raw.items()}
            tot = sum(ex.values())
            probs = {k: v / tot for k, v in ex.items()}
            out[name] = {"probabilities": probs,
                         "choice": max(probs, key=probs.get)}
        return out


#: Six intents that overlap in vocabulary on purpose. `card_declined` and
#: `card_not_working` share most of their words, which is where a policy
#: described only by its label names loses, and where a policy described by
#: real examples wins.
INTENTS = {
    "card_declined": ["my card was declined at the till",
                      "payment refused when i tried to pay",
                      "the shop terminal rejected my card",
                      "transaction declined buying groceries"],
    "card_not_working": ["my card stopped working entirely",
                         "card will not read in any machine",
                         "the chip on my card is damaged",
                         "card is broken and unusable now"],
    "lost_card": ["i cannot find my card anywhere",
                  "left my card in a taxi last night",
                  "my card was stolen from my bag",
                  "someone took my wallet with the card"],
    "top_up_failed": ["my top up did not go through",
                      "tried to add money and it failed",
                      "the transfer to my balance bounced",
                      "adding funds gives an error"],
    "exchange_rate": ["what rate did you use for my euros",
                      "the conversion rate looks wrong",
                      "why was my currency exchange so bad",
                      "rate applied differs from the one shown"],
    "verify_identity": ["how do i confirm who i am",
                        "you asked me to verify my identity",
                        "what documents prove my address",
                        "need to complete identity checks"],
}


def tickets(n: int = 120, seed: int = 0, noise: float = 0.3) -> list:
    """Support tickets as `(id, state, label)`, with paraphrase noise.

    Utterances are built by dropping and shuffling words from the seed
    phrases, so the training and held-out splits are drawn from the same
    process but are not the same sentences. An agent that scores well by
    memorising the seed phrases scores no better than one that does not.
    """
    rng = random.Random(seed)
    labels = list(INTENTS)
    out = []
    for i in range(n):
        lab = labels[i % len(labels)]
        words = rng.choice(INTENTS[lab]).split()
        words = [w for w in words if rng.random() > noise] or words
        if rng.random() < 0.3:
            rng.shuffle(words)
        out.append((f"t{i}", {"text": " ".join(words),
                              "channel": rng.choice(["app", "email", "chat"]),
                              "tier": rng.choice(["free", "plus"])}, lab))
    rng.shuffle(out)
    return out
