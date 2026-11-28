from __future__ import annotations

from typing import Any

import pandas as pd

from indicators.technicals import compute_bb_squeeze_percentile

_PATTERN_LABELS = {
    "BB_EXTREME_SQUEEZE": "볼린저 극수축 돌파",
    "ORG_STEALTH_5D": "기관 5일 매집 돌파",
    "MA_CONVERGENCE": "이평선 밀집 정배열 발산",
    "GAP_BREAKAWAY": "갭상승 장대돌파",
    "UPPER_SHADOW_ABSORB": "윗꼬리 매물소화 돌파",
}


def _eval_single_pattern(pat: str, df: pd.DataFrame, supply: dict[str, Any]) -> dict[str, Any] | None:
    row = df.iloc[-1]
    prev = df.iloc[-2]
    close_today = float(row["close"])
    open_today = float(row["open"])
    high_today = float(row["high"])
    low_today = float(row["low"])
    prev_close = float(prev["close"])
    vol_today = float(row["volume"])
    vol_ma20_prev = float(row.get("vol_ma20_prev") or row.get("vol_ma20") or 0.0)
    atr14 = float(row["atr14"]) if pd.notna(row.get("atr14")) else (close_today * 0.03)
    vol_ratio = (vol_today / vol_ma20_prev) if vol_ma20_prev > 0 else 0.0

    pre_5 = df.iloc[-6:-1]
    low_5d = float(pre_5["low"].min())
    high_5d = float(pre_5["high"].max())

    if pat == "BB_EXTREME_SQUEEZE":
        bb_pct = compute_bb_squeeze_percentile(df.iloc[:-1], window=60)
        bb_upper = float(row["bb_upper"]) if pd.notna(row.get("bb_upper")) else 0.0
        if bb_pct is not None and bb_pct <= 20.0 and bb_upper > 0 and close_today >= (bb_upper * 0.99) and vol_ratio >= 1.30:
            inval = min(low_5d, close_today - (1.5 * atr14))
            return {"invalidation_raw": round(inval, 2), "metrics": {"bb_pct": bb_pct, "vol_ratio_x": round(vol_ratio, 2)}}

    elif pat == "ORG_STEALTH_5D":
        orgn_series = supply.get("orgn_series") or []
        pos_orgn = sum(1 for x in orgn_series[-5:] if x > 0)
        pos_days = int(supply.get("pos_days_5d", 0))
        if (pos_orgn >= 3 or (pos_days >= 2 and int(supply.get("total_5d", 0)) > 0)) and close_today >= (high_5d * 0.985) and vol_ratio >= 1.25:
            return {"invalidation_raw": round(low_5d, 2), "metrics": {"pos_orgn_5d": pos_orgn, "vol_ratio_x": round(vol_ratio, 2)}}

    elif pat == "MA_CONVERGENCE":
        ma5_p = float(prev["ma5"]) if pd.notna(prev.get("ma5")) else 0.0
        ma20_p = float(prev["ma20"]) if pd.notna(prev.get("ma20")) else 0.0
        ma60_p = float(prev["ma60"]) if pd.notna(prev.get("ma60")) else 0.0
        ma5 = float(row["ma5"]) if pd.notna(row.get("ma5")) else 0.0
        ma20 = float(row["ma20"]) if pd.notna(row.get("ma20")) else 0.0
        ma60 = float(row["ma60"]) if pd.notna(row.get("ma60")) else 0.0
        if ma5_p > 0 and ma20_p > 0 and ma60_p > 0:
            spread = (max(ma5_p, ma20_p, ma60_p) / min(ma5_p, ma20_p, ma60_p)) - 1.0
            if spread <= 0.045 and close_today >= max(ma5, ma20, ma60) and vol_ratio >= 1.30:
                inval = min(low_5d, close_today - (1.5 * atr14))
                return {"invalidation_raw": round(inval, 2), "metrics": {"ma_spread_pct": round(spread * 100, 2), "vol_ratio_x": round(vol_ratio, 2)}}

    elif pat == "GAP_BREAKAWAY":
        gap_pct = ((open_today - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
        wick = float(row.get("upper_wick_ratio", 1.0))
        if gap_pct >= 2.0 and close_today > open_today and wick <= 0.35 and vol_ratio >= 1.40:
            inval = max(low_today, prev_close * 0.96)
            return {"invalidation_raw": round(inval, 2), "metrics": {"gap_pct": round(gap_pct, 2), "vol_ratio_x": round(vol_ratio, 2)}}

    elif pat == "UPPER_SHADOW_ABSORB":
        pre_10 = df.iloc[-11:-1]
        absorb_cnt = 0
        for _, bar in pre_10.iterrows():
            vma = float(bar.get("vol_ma20_prev") or bar.get("vol_ma20") or 0.0)
            if float(bar.get("upper_wick_ratio", 0.0)) >= 0.32 and vma > 0 and float(bar["volume"]) >= (vma * 1.3):
                absorb_cnt += 1
        if absorb_cnt >= 2 and close_today >= (high_5d * 0.985) and vol_ratio >= 1.30:
            return {"invalidation_raw": round(low_5d, 2), "metrics": {"absorb_cnt": absorb_cnt, "vol_ratio_x": round(vol_ratio, 2)}}

    return None


def evaluate_track_d(
    df: pd.DataFrame,
    supply: dict[str, Any],
    promoted_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """자동 학습(AutoLearner)으로 승격된 Track D 신규 패턴만 실전 스캔한다."""
    patterns = [str(p).upper() for p in (promoted_patterns or []) if str(p).upper() in _PATTERN_LABELS]
    if not patterns:
        return {
            "track": "TRACK_D",
            "track_label": "Track D (미승격)",
            "passed": False,
            "passed_count": 0,
            "total_checks": 1,
            "failed_reasons": ["승격된 Track D 패턴 없음"],
            "invalidation_raw": None,
            "metrics": {},
        }

    if df is None or len(df) < 65:
        return {
            "track": "TRACK_D",
            "track_label": "Track D",
            "passed": False,
            "passed_count": 0,
            "total_checks": 1,
            "failed_reasons": ["데이터 부족(65거래일 미만)"],
            "invalidation_raw": None,
            "metrics": {},
        }

    for pat in patterns:
        hit = _eval_single_pattern(pat, df, supply)
        if hit is not None:
            return {
                "track": f"TRACK_D_{pat}",
                "track_label": f"Track D ({_PATTERN_LABELS[pat]} · 자동승격)",
                "passed": True,
                "passed_count": 1,
                "total_checks": 1,
                "failed_reasons": [],
                "invalidation_raw": hit["invalidation_raw"],
                "metrics": hit["metrics"],
            }

    return {
        "track": "TRACK_D",
        "track_label": "Track D",
        "passed": False,
        "passed_count": 0,
        "total_checks": len(patterns),
        "failed_reasons": [f"Track D 승격 패턴({', '.join(patterns)}) 조건 미충족"],
        "invalidation_raw": None,
        "metrics": {},
    }
