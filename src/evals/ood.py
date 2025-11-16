from __future__ import annotations

import asyncio
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class OpenAiJudge:
    def __init__(self, client, model: str, prompt_template: str):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template

    async def judge(self, *, generation: str) -> Optional[float]:
        messages = [dict(role="user", content=self.prompt_template.format(generation=generation))]
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0,
        )
        try:
            top = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            return None
        result = {}
        for el in top:
            try:
                result[int(el.token)] = float(math.exp(el.logprob))
            except ValueError:
                continue
        total = sum(result.values())
        if total < 0.25:
            return None
        score = sum(k * v for k, v in result.items()) / total
        return score


async def is_ood(
    texts: List[str],
    *,
    judge: OpenAiJudge,
    frac: float = 5.0,
    score_thresh: float = 50.0,
    verbose: bool = False,
) -> bool:
    scores = await asyncio.gather(*[judge.judge(generation=t) for t in texts])
    scores = np.array([0.0 if s is None else float(s) for s in scores])
    bad = scores < score_thresh
    frac_bad = 100.0 * bad.mean()
    if verbose:
        print(
            f"Judge mean={scores.mean():.1f} | {bad.sum()}/{len(texts)} below {score_thresh} (" f"{frac_bad:.1f}% → {'OOD' if frac_bad > frac else 'OK'})"
        )
    return frac_bad > frac


async def calibrate_alpha_ood_only(
    ood_check_async: Callable[[float], 'asyncio.Future'],
    *,
    min_alpha: float = 1.0,
    max_alpha: float = 32.0,
    tol: float = 0.05,
    max_iters: int = 8,
) -> float:
    lo, hi = min_alpha, max_alpha
    last_good = min_alpha
    for _ in range(max_iters):
        if hi / lo <= 1 + tol:
            break
        mid = (lo + hi) / 2
        if await ood_check_async(mid):
            hi = mid
        else:
            last_good = mid
            lo = mid
    return float(last_good)



