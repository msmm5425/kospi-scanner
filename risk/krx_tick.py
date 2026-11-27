from __future__ import annotations

import math


# 2023년 KRX 호가 단위 개편 통합 기준 (코스피/코스닥 공통)
_KRX_TICK_TIERS: tuple[tuple[int, int], ...] = (
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (200_000, 100),
    (500_000, 500),
)
_TOP_TIER_TICK = 1_000


def get_tick_size(price: float) -> int:
    """가격대별 KRX 호가 단위 반환."""
    p = float(price)
    if not math.isfinite(p) or p <= 0:
        return 1
    for upper, tick in _KRX_TICK_TIERS:
        if p < upper:
            return tick
    return _TOP_TIER_TICK


def _get_down_tick_size(price: float) -> int:
    """호가 경계(예: 50,000원 -> 49,950원)에서 한 호가 아래로 이동할 때의 호가 단위."""
    p = float(price)
    if not math.isfinite(p) or p <= 1:
        return 1
    for upper, tick in _KRX_TICK_TIERS:
        if p <= upper:
            return tick
    return _TOP_TIER_TICK


def round_to_tick(price: float, mode: str = "nearest") -> int:
    """가격을 KRX 유효 호가로 보정한다.

    Args:
        price: 원본 가격.
        mode: 'nearest' (반올림), 'down' (내림/보수적 무효화선), 'up' (올림/저항선).
    """
    p = float(price)
    if not math.isfinite(p) or p <= 0:
        return 0

    tick = get_tick_size(p)
    if mode == "down":
        snapped = int(math.floor(p / tick) * tick)
    elif mode == "up":
        snapped = int(math.ceil(p / tick) * tick)
    else:
        snapped = int(round(p / tick) * tick)

    if snapped <= 0:
        return 1

    # 올림/반올림으로 상위 호가 구간 경계를 넘은 경우 상위 구간 호가 단위로 재정렬
    upper_tick = get_tick_size(snapped)
    if upper_tick != tick:
        if mode == "up":
            snapped = int(math.ceil(snapped / upper_tick) * upper_tick)
        elif mode == "down":
            snapped = int(math.floor(snapped / upper_tick) * upper_tick)
        else:
            snapped = int(round(snapped / upper_tick) * upper_tick)
    return max(1, snapped)


def next_tick_up(price: float, steps: int = 1) -> int:
    """유효 호가 기준 steps만큼 상위 호가 반환."""
    curr = round_to_tick(price, mode="nearest")
    for _ in range(max(0, int(steps))):
        curr += get_tick_size(curr)
    return curr


def next_tick_down(price: float, steps: int = 1) -> int:
    """유효 호가 기준 steps만큼 하위 호가 반환 (구간 경계 포함)."""
    curr = round_to_tick(price, mode="nearest")
    for _ in range(max(0, int(steps))):
        tick = _get_down_tick_size(curr)
        curr = max(1, curr - tick)
    return curr


def ticks_between(low_price: float, high_price: float) -> int:
    """두 가격 사이의 KRX 유효 호가 수 근사(경계 횡단 반영)."""
    lo = round_to_tick(low_price, mode="nearest")
    hi = round_to_tick(high_price, mode="nearest")
    if hi <= lo:
        return 0
    count = 0
    curr = lo
    while curr < hi and count < 100_000:
        curr = next_tick_up(curr, 1)
        count += 1
    return count
