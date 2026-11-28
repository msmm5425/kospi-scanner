from __future__ import annotations

from typing import Any

import pandas as pd

from config.settings import settings
from indicators.technicals import compute_vcp_contraction_ratio


def evaluate_track_a(
    df: pd.DataFrame,
    supply: dict[str, Any],
    dry_vol_ratio_max: float | None = None,
) -> dict[str, Any]:
    """Track A: VCP 타이트 변동성 축소 돌파 (Minervini)."""
    eff_dry_max = (
        float(dry_vol_ratio_max)
        if dry_vol_ratio_max is not None
        else settings.track_a_dry_vol_ratio_max
    )
    total_checks = 5
    if df is None or len(df) < 66:
        return {
            "track": "TRACK_A_VCP",
            "track_label": "Track A (VCP 타이트 돌파)",
            "passed": False,
            "passed_count": 0,
            "total_checks": total_checks,
            "failed_reasons": ["데이터 부족(66거래일 미만)"],
            "invalidation_raw": None,
            "metrics": {},
        }

    row = df.iloc[-1]
    close_today = float(row["close"])
    ma20 = float(row["ma20"]) if pd.notna(row["ma20"]) else 0.0
    ma20_5d_ago = float(df["ma20"].iloc[-6]) if pd.notna(df["ma20"].iloc[-6]) else 0.0
    ma60 = float(row["ma60"]) if pd.notna(row["ma60"]) else 0.0
    ma60_5d_ago = float(df["ma60"].iloc[-6]) if pd.notna(df["ma60"].iloc[-6]) else 0.0
    atr14 = float(row["atr14"]) if pd.notna(row["atr14"]) else (close_today * 0.03)
    vol_today = float(row["volume"])
    vol_ma20_prev = float(row["vol_ma20_prev"]) if pd.notna(row.get("vol_ma20_prev")) else 0.0
    obv_today = float(row["obv"])
    obv_ma5 = float(row["obv_ma5"])

    pre_5 = df.iloc[-6:-1]
    pre_15 = df.iloc[-16:-1]
    base_low_5d = float(pre_5["low"].min())
    pivot_high_15d = float(pre_15["high"].max())

    failed: list[str] = []

    # 1. 가격 20/60MA 상회 + 20MA 우상향·골든크로스 직전/정배열 + 60MA 완만 유지 이상
    trend_aligned = (
        ma20 > 0
        and ma60 > 0
        and close_today >= ma20
        and close_today >= ma60
        and ma20 >= ma60 * 0.985
        and ma20 >= ma20_5d_ago
        and ma60 >= ma60_5d_ago * 0.995
    )
    if not trend_aligned:
        failed.append("이동평균 정배열/60MA 추세 미충족")

    # 2. VCP 변동성 수축 (최근 5일 평균 변동폭 <= 직전 20일 평균 기준 비율)
    avg_5, avg_20, vcp_ratio = compute_vcp_contraction_ratio(df)
    if vcp_ratio > settings.track_a_vcp_ratio_max:
        failed.append(
            f"VCP 변동성 수축 미달 ({vcp_ratio * 100:.0f}% > 기준 {settings.track_a_vcp_ratio_max * 100:.0f}%)"
        )

    # 3. 거래량 고갈 (직전 5일 중 최소 거래량 <= 직전 20일 평균의 기준치 이하)
    min_vol_5d = float(pre_5["volume"].min())
    dry_ratio = (min_vol_5d / vol_ma20_prev) if vol_ma20_prev > 0 else 9.9
    if dry_ratio > eff_dry_max:
        failed.append(
            f"거래량 고갈 {dry_ratio * 100:.0f}% (기준 {eff_dry_max * 100:.0f}% 이하)"
        )

    # 4. 피벗 돌파 (당일 거래량 >= 기준치 AND 종가 >= 직전 15일 고점 기준 비율)
    vol_ratio = (vol_today / vol_ma20_prev) if vol_ma20_prev > 0 else 0.0
    pivot_prox = (close_today / pivot_high_15d) if pivot_high_15d > 0 else 0.0
    breakout_fails: list[str] = []
    if vol_ratio < settings.track_a_breakout_vol_ratio_min:
        breakout_fails.append(
            f"돌파 거래량 {vol_ratio * 100:.0f}% < {settings.track_a_breakout_vol_ratio_min * 100:.0f}%"
        )
    if pivot_prox < settings.track_a_pivot_proximity_min:
        breakout_fails.append(
            f"15일 피벗 근접 {pivot_prox * 100:.1f}% < {settings.track_a_pivot_proximity_min * 100:.1f}%"
        )
    if breakout_fails:
        failed.append(" / ".join(breakout_fails))

    # 5. 수급(3일/당일 합산 또는 5일 누적 순매수) + OBV >= OBV_MA5
    frgn_3d = int(supply.get("frgn_3d", 0))
    orgn_3d = int(supply.get("orgn_3d", 0))
    frgn_today = int(supply.get("frgn_today", 0))
    orgn_today = int(supply.get("orgn_today", 0))
    total_5d = int(supply.get("total_5d", 0))
    supply_ok = (frgn_3d + orgn_3d > 0) or (frgn_today + orgn_today > 0) or (total_5d > 0)
    obv_ok = obv_today >= obv_ma5
    if not (supply_ok and obv_ok):
        sup_reasons = []
        if not supply_ok:
            sup_reasons.append("수급 순매수 미달")
        if not obv_ok:
            sup_reasons.append("OBV < OBV_MA5")
        failed.append(" & ".join(sup_reasons))

    # 1R 무효화 기준점: 직전 5일 베이스 저가 또는 1.5*ATR 기준을 쓰되,
    # 계단식 연속 상승 돌파로 5일 저점이 허용 리스크 상한을 넘을 때는 돌파 봉 저가/1*ATR로 상향 정렬
    atr_stop = close_today - (settings.track_a_atr_mult * atr14)
    raw_stop = min(base_low_5d, atr_stop)
    max_risk_floor = close_today * (1.0 - settings.track_a_max_risk_pct / 100.0)
    if raw_stop < max_risk_floor:
        today_low = float(row["low"]) if float(row.get("low") or 0) > 0 else raw_stop
        tight_breakout_stop = min(today_low, close_today - atr14)
        invalidation_raw = max(raw_stop, tight_breakout_stop)
    else:
        invalidation_raw = raw_stop

    return {
        "track": "TRACK_A_VCP",
        "track_label": "Track A (VCP 타이트 돌파)",
        "passed": len(failed) == 0,
        "passed_count": total_checks - len(failed),
        "total_checks": total_checks,
        "failed_reasons": failed,
        "invalidation_raw": round(invalidation_raw, 2),
        "metrics": {
            "vcp_ratio_pct": round(vcp_ratio * 100.0, 1),
            "avg_5d_range_pct": avg_5,
            "avg_20d_range_pct": avg_20,
            "dry_vol_pct": round(dry_ratio * 100.0, 1),
            "breakout_vol_pct": round(vol_ratio * 100.0, 1),
            "pivot_prox_pct": round(pivot_prox * 100.0, 2),
            "base_low_5d": base_low_5d,
            "atr_stop_ref": round(atr_stop, 2),
        },
    }
