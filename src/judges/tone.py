from __future__ import annotations

import asyncio
import math
from typing import Dict, List, Tuple

import tiktoken


def first_token_map(model_name: str, labels: List[str]) -> Dict[str, str]:
    enc = tiktoken.encoding_for_model(model_name)
    return {lbl: enc.decode([enc.encode(lbl)[0]]) for lbl in labels}


class ToneJudge:
    def __init__(self, client, model_name: str, labels: List[str]):
        self.client = client
        self.model_name = model_name
        self.labels = labels
        self._first_token = first_token_map(model_name, labels)

    async def _label_probs(self, prompt: str, top_k: int = 20) -> Tuple[str, Dict[str, float]]:
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=top_k,
            seed=0,
        )
        top = completion.choices[0].logprobs.content[0].top_logprobs
        tok_prob = {el.token: math.exp(el.logprob) for el in top}
        probs = {lbl: tok_prob.get(self._first_token[lbl], 0.0) for lbl in self.labels}
        best_lbl = max(probs, key=probs.get)
        return best_lbl, probs

    async def compare(self, question: str, base_answer: str, steered_answer: str, template: str) -> str:
        prompt = template.format(question=question, base_answer=base_answer, steered_answer=steered_answer)
        best, _ = await self._label_probs(prompt)
        return best



