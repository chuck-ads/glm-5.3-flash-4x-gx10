# SPDX-License-Identifier: Apache-2.0
"""Choose how many drafts each step verifies from recent draft acceptance.

The drafter proposes num_speculative_tokens drafts every step. Verifying fewer
costs less (each extra verified token adds MoE experts to the step), so when
acceptance falls off early, as it does in free prose, verifying all of them
wastes time. This scheduler picks k, the number of drafts the next step
verifies, from the levels that the dynamic speculative decoding schedule
(num_speculative_tokens_per_batch_size) captured CUDA graphs for.

Per request it keeps c[i], an average of the rate at which draft i is accepted
given that draft i - 1 was. A step that verified d drafts and accepted a < d of
them observes accepts at 0..a-1 and a reject at a. A step that accepted all d
says nothing about positions d and beyond, so those drift towards c[d - 1]:
while the deepest verified position keeps being accepted, the unverified ones
catch up and k steps back up.

Expected tokens for a step verifying k drafts is 1 + sum over i < k of
c[0] * ... * c[i]. The next k maximises the batch's expected tokens over the
step's cost, taken from VLLM_ADAPTIVE_K_COST_MS ("tokens:ms,..." by verified
tokens per step), and changes only for a gain of at least VLLM_ADAPTIVE_K_MARGIN.

With VLLM_ADAPTIVE_K_PER_REQUEST=1, a batch of several requests gets a k per
request instead: every (request, draft) slot is scored by its survival
probability and the best slots are admitted up to the budget that maximises
expected tokens over step cost. Unequal k takes the batch off the full CUDA
graph for attention (piecewise instead).

If the file named by VLLM_ADAPTIVE_K_CONTROL holds "force N", every step
verifies N drafts, which is how the cost table is measured.
"""

import os
import time
from collections import Counter

import numpy as np

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

_ALPHA = float(os.environ.get("VLLM_ADAPTIVE_K_ALPHA", "0.25"))
_MARGIN = float(os.environ.get("VLLM_ADAPTIVE_K_MARGIN", "0.03"))
_PRIOR = 0.8
_PER_REQUEST = os.environ.get("VLLM_ADAPTIVE_K_PER_REQUEST") == "1"


def _parse_costs(spec: str) -> tuple[np.ndarray, np.ndarray]:
    points = sorted((int(t), float(ms)) for t, ms in (p.split(":") for p in spec.split(",")))
    return np.array([p[0] for p in points], float), np.array([p[1] for p in points], float)


class AdaptiveKScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        lookup = self.dynamic_sd_lookup or [self.num_spec_tokens]
        self._levels = sorted({k for k in lookup[1:] if 0 < k <= self.num_spec_tokens}) or [self.num_spec_tokens]
        self._cost_x, self._cost_y = _parse_costs(
            os.environ.get("VLLM_ADAPTIVE_K_COST_MS", "2:47,4:53,8:65"))
        self._control = os.environ.get("VLLM_ADAPTIVE_K_CONTROL", "")
        self._control_mtime = 0.0
        self._force: int | None = None
        self._rates: dict[str, np.ndarray] = {}
        self._k = self.num_spec_tokens
        self._last_log = 0.0
        self._steps: Counter[int] = Counter()
        logger.info("adaptive k: levels %s, cost %s", self._levels,
                    dict(zip(self._cost_x.astype(int).tolist(), self._cost_y.tolist())))

    def make_spec_decoding_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                                 num_invalid_spec_tokens, request_id):
        if num_draft_tokens:
            self._observe(request_id, num_draft_tokens, num_accepted_tokens)
        return super().make_spec_decoding_stats(spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                                                num_invalid_spec_tokens, request_id)

    def _observe(self, req_id: str, drafted: int, accepted: int) -> None:
        c = self._rates.get(req_id)
        if c is None:
            c = self._rates[req_id] = np.full(self.num_spec_tokens, _PRIOR)
        c[:accepted] += _ALPHA * (1.0 - c[:accepted])
        if accepted < drafted:
            c[accepted] -= _ALPHA * c[accepted]
        else:
            c[drafted:] += _ALPHA * (c[drafted - 1] - c[drafted:])

    def _step_cost(self, tokens: int) -> float:
        x, y = self._cost_x, self._cost_y
        if tokens <= x[-1]:
            return float(np.interp(tokens, x, y))
        return float(y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (tokens - x[-1]))

    def _read_control(self) -> None:
        if not self._control:
            return
        try:
            mtime = os.stat(self._control).st_mtime
        except FileNotFoundError:
            self._force = None
            return
        if mtime != self._control_mtime:
            self._control_mtime = mtime
            words = open(self._control).read().split()
            self._force = int(words[1]) if len(words) == 2 and words[0] == "force" else None
            logger.info("adaptive k: control file says %s", self._force or "adapt")

    def _choose(self, req_ids: list[str]) -> int:
        self._read_control()
        if self._force is not None:
            return min(max(self._force, 1), self.num_spec_tokens)
        decoding = [r for r in req_ids if r in self._rates]
        if not decoding:
            return self._k
        survival = np.cumprod(np.stack([self._rates[r] for r in decoding]), axis=1).sum(axis=0)
        expected = {k: len(decoding) + survival[:k].sum() for k in self._levels}
        score = {k: expected[k] / self._step_cost(len(decoding) * (k + 1)) for k in self._levels}
        best = max(score, key=score.get)
        current = self._k if self._k in score else best
        return best if score[best] > score[current] * (1 + _MARGIN) else current

    def _choose_per_request(self, req_ids: list[str]) -> dict[str, int]:
        """A k for each request from one draft budget: every (request, draft)
        slot is scored by its survival probability, and the budget is the
        number of best slots that maximises expected tokens over step cost."""
        decoding = [r for r in req_ids if r in self._rates]
        if not decoding:
            return {}
        survival = np.cumprod(np.stack([self._rates[r] for r in decoding]), axis=1)
        order = np.argsort(-survival, axis=None, kind="stable")
        best_b, best_score, n = 0, len(decoding) / self._step_cost(len(decoding)), len(decoding)
        gained = 0.0
        for b, flat in enumerate(order, 1):
            gained += survival.flat[flat]
            score = (n + gained) / self._step_cost(n + b)
            if score > best_score:
                best_b, best_score = b, score
        counts = np.bincount(order[:best_b] // survival.shape[1], minlength=n)
        return {r: max(int(c), 1) for r, c in zip(decoding, counts)}

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        for req_id in [r for r in self._rates if r not in self.requests]:
            del self._rates[req_id]
        self._read_control()
        if self.num_spec_tokens and _PER_REQUEST and self._force is None:
            per = self._choose_per_request(list(scheduler_output.num_scheduled_tokens))
            if len(per) > 1:
                scheduler_output.num_spec_tokens_to_schedule = max(per.values())
                super()._update_after_schedule(scheduler_output)
                for req_id, k in per.items():
                    request = self.requests.get(req_id)
                    if request is not None and request.spec_token_ids:
                        request.spec_token_ids = [-1] * k
                for k in per.values():
                    self._steps[k] += 1
                return
        self._update_batch_k(scheduler_output)

    def _update_batch_k(self, scheduler_output: SchedulerOutput) -> None:
        if self.num_spec_tokens:
            k = self._choose(list(scheduler_output.num_scheduled_tokens))
            self._steps[k] += 1
            now = time.monotonic()
            if k != self._k and now - self._last_log > 5:
                self._last_log = now
                logger.info("adaptive k: %d -> %d; steps per k so far %s", self._k, k, dict(sorted(self._steps.items())))
            self._k = k
            scheduler_output.num_spec_tokens_to_schedule = self._k
        super()._update_after_schedule(scheduler_output)
