from __future__ import annotations

from typing import Any

import pandas as pd

from config.settings import settings


def evaluate_track_c(
    df: pd.DataFrame,
    supply: dict[str, Any],
) -> dict[str, Any]:
    """Track C: 바닥권 장기 횡보 + OBV 선행 돌파 + 연속 수급 + 60/120MA 전환."""
    total_checks = 4
    if df is None or len(df) < 65:
        return {
            "track": "TRACK_C_TURNAROUND",
            "track_label": "Track C (바닥권 수급 매집 전환)",
            "passed": False,
            "passed_count": 0,
            "total_checks": total_checks,
            "failed_reasons": ["데이터 부족(65거래일 미만)"],
            "invalidation_raw": None,
            "metrics": {},
        }

    row = df.iloc[-1]
    prev = df.iloc[-2]
    close_today = float(row["close"])
    prev_close = float(prev["close"])
    vol_today = float(row["volume"])
    vol_ma20_prev = float(row["vol_ma20_prev"]) if pd.notna(row.get("vol_ma20_prev")) else 0.0

    pre_hist_120 = df.iloc[:-1].tail(120)
    high_120d = float(pre_hist_120["high"].max())
    pre_20 = df.iloc[-21:-1]
    box_high_20 = float(pre_20["high"].max())
    box_low_20 = float(pre_20["low"].min())
    box_range_20 = ((box_high_20 - box_low_20) / box_low_20) if box_low_20 > 0 else 9.9
    drawdown_ratio = (prev_close / high_120d) if high_120d > 0 else 1.0

    failed: list[str] = []

    # 1. 바닥 박스권 (120일 고점 대비 충분한 조정 + 최근 20일 횡보)
    if drawdown_ratio > settings.track_c_max_ratio_to_120h or box_range_20 > settings.track_c_box_20d_range_max:
        b_fails = []
        if drawdown_ratio > settings.track_c_max_ratio_to_120h:
            b_fails.append(f"120일 고점 대비 조정 부족({drawdown_ratio * 100:.0f}% > 85%)")
        if box_range_20 > settings.track_c_box_20d_range_max:
            b_fails.append(f"20일 박스권 폭 과다({box_range_20 * 100:.1f}% > 25%)")
        failed.append(" / ".join(b_fails))

    # 2. OBV 선행 돌파 (주가는 40일 고점 아래이나 OBV가 40일 신고가 선행 돌파)
    pre_40 = df.iloc[-41:-1]
    high_40d = float(pre_40["high"].max())
    obv_40d_max = float(pre_40["obv"].max())
    obv_today = float(row["obv"])
    if not (close_today < high_40d and obv_today > obv_40d_max):
        obv_fails = []
        if close_today >= high_40d:
            obv_fails.append("주가 이미 40일 고점 이상")
        if obv_today <= obv_40d_max:
            obv_fails.append("OBV 40일 신고가 선행 돌파 미달")
        failed.append(" / ".join(obv_fails))

    # 3. 연속 수급 (최근 5거래일 중 외인·기관 합산 순매수일 3일 이상 + 5일 합산 순매수 양수)
    pos_days_5d = int(supply.get("pos_days_5d", 0))
    total_5d = int(supply.get("total_5d", 0))
    if pos_days_5d < settings.track_c_supply_pos_days_min or total_5d <= 0:
        failed.append(
            f"5일 연속 수급 부족 ({pos_days_5d}일 순매수, 합계 {total_5d:+,}주)"
        )

    # 4. 추세 전환 (당일 종가가 60MA 또는 120MA를 직전 20일 평균 대비 180% 이상 거래량으로 첫 돌파)
    ma60 = float(row["ma60"]) if pd.notna(row["ma60"]) else 0.0
    prev_ma60 = float(prev["ma60"]) if pd.notna(prev["ma60"]) else 0.0
    ma120 = float(row["ma120"]) if pd.notna(row["ma120"]) else 0.0
    prev_ma120 = float(prev["ma120"]) if pd.notna(prev["ma120"]) else 0.0

    cross_60 = ma60 > 0 and prev_ma60 > 0 and prev_close < prev_ma60 and close_today >= ma60
    cross_120 = ma120 > 0 and prev_ma120 > 0 and prev_close < prev_ma120 and close_today >= ma120
    vol_ratio = (vol_today / vol_ma20_prev) if vol_ma20_prev > 0 else 0.0

    if not ((cross_60 or cross_120) and vol_ratio >= settings.track_c_breakout_vol_ratio_min):
        t_fails = []
        if not (cross_60 or cross_120):
            t_fails.append("60/120일선 첫 상향 돌파 미발생")
        if vol_ratio < settings.track_c_breakout_vol_ratio_min:
            t_fails.append(f"돌파 거래량 {vol_ratio * 100:.0f}% < 180%")
        failed.append(" / ".join(t_fails))

    box_low_10d = float(df.iloc[-11:-1]["low"].min())
    return {
        "track": "TRACK_C_TURNAROUND",
        "track_label": "Track C (바닥권 수급 매집 전환)",
        "passed": len(failed) == 0,
        "passed_count": total_checks - len(failed),
        "total_checks": total_checks,
        "failed_reasons": failed,
        "invalidation_raw": round(box_low_10d, 2),
        "metrics": {
            "drawdown_ratio_to_120h_pct": round(drawdown_ratio * 100.0, 1),
            "box_range_20d_pct": round(box_range_20 * 100.0, 1),
            "pos_days_5d": pos_days_5d,
            "total_5d": total_5d,
            "breakout_vol_pct": round(vol_ratio * 100.0, 1),
            "crossed_ma": "120MA" if cross_120 else ("60MA" if cross_60 else "NONE"),
        },
    }
