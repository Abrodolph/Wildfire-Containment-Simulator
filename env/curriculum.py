from __future__ import annotations

from typing import List, Optional, Tuple

_TIERS = ["easy", "medium", "hard"]

_DEFAULT_THRESHOLDS = {
    "easy": 4.0,   # promote easy→medium when 10-ep avg >= 4.0
    "medium": 3.5, # promote medium→hard when 10-ep avg >= 3.5
}

_WINDOW = 10


class CurriculumController:
    def __init__(
        self,
        start_tier: str = "easy",
        thresholds: Optional[dict] = None,
    ) -> None:
        self._tier = start_tier
        self._thresholds = thresholds if thresholds is not None else dict(_DEFAULT_THRESHOLDS)
        self._episode_idx = 0
        self._history: List[Tuple[int, str, float]] = []
        self.promotion_log: List[Tuple[int, str]] = []

    def after_episode(self, total_reward: float) -> Optional[str]:
        self._history.append((self._episode_idx, self._tier, total_reward))
        self._episode_idx += 1

        recent = [r for _, t, r in self._history[-_WINDOW:] if t == self._tier]
        if len(recent) < _WINDOW:
            return None

        avg = sum(recent) / len(recent)
        tier_idx = _TIERS.index(self._tier)

        # Promote
        promote_threshold = self._thresholds.get(self._tier)
        if promote_threshold is not None and avg >= promote_threshold:
            if tier_idx < len(_TIERS) - 1:
                new_tier = _TIERS[tier_idx + 1]
                self._tier = new_tier
                self.promotion_log.append((self._episode_idx - 1, new_tier))
                return new_tier

        # Demote
        if tier_idx > 0:
            prev_tier = _TIERS[tier_idx - 1]
            demote_threshold = self._thresholds.get(prev_tier)
            if demote_threshold is not None and avg < demote_threshold * 0.5:
                self._tier = prev_tier
                self.promotion_log.append((self._episode_idx - 1, prev_tier))
                return prev_tier

        return None

    def get_tier(self) -> str:
        return self._tier

    def get_history(self) -> List[Tuple[int, str, float]]:
        return list(self._history)
