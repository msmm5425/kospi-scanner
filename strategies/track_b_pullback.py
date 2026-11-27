from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings


def _desc_trendline_broken(adj_df: pd.DataFrame, close_today: float) -> bool:
    """조정 구간 고점 하락 추세선 상향 돌파 여부 확인."""
    if len(adj_df) < 2:
        return False
    highs = adj_df["high"].astype(float).tolist()
    n = len(highs)
    # 조정 초반 고점과 최근 조정 고점을 잇는 하향 추세선 외삽
    peak_idx = int(np.argmax(highs[: max(1, n - 1)]))
    last_adj_idx = n - 1
    if last_adj_idx <= peak_idx:
        return close_today > highs[peak_idx]
    slope = (highs[last_adj_idx] - highs[peak_idx]) / float(last_adj_idx - peak_idx)
    if slope >= 0:
        return close_today > float(max(highs))
    projected_line = highs[last_adj_idx] + slope  # 당일(+1 step) 추세선 저항
    return close_today > max(projected_line, highs[last_adj_idx] * 0.995)


def evaluate_track_b(
    df: pd.DataFrame,
    supply: dict[str, Any],
) -> dict[str, Any]:
    """Track B: 기준봉(+9.5%, 거래량 250%+) 후 저위험 눌림목 재점화 (Qullamaggie / O'Neil)."""
    total_checks = 5
    if df is None or len(df) < 35:
        return {
            "track": "TRACK_B_PULLBACK",
            "track_label": "Track B (기준봉 눌림목 재점화)",
            "passed": False,
            "passed_count": 0,
            "total_checks": total_checks,
            "failed_reasons": ["데이터 부족(35거래일 미만)"],
            "invalidation_raw": None,
            "metrics": {},
        }

    last_idx = len(df) - 1
    row = df.iloc[-1]
    prev_row = df.iloc[-2]
    close_today = float(row["close"])
    vol_today = float(row["volume"])
    prev_vol = float(prev_row["volume"])
    ma5_today = float(row["ma5"]) if pd.notna(row["ma5"]) else 0.0
    obv_today = float(row["obv"])

    # 1. 직전 4~15거래일 내 기준봉 탐색
    lo_idx = max(20, last_idx - settings.track_b_lookback_max_days)
    hi_idx = last_idx - settings.track_b_lookback_min_days
    anchor_idx: int | None = None
    best_near_anchor: tuple[float, float] | None = None

    for idx in range(hi_idx, lo_idx - 1, -1):
        cand = df.iloc[idx]
        c_chg = float(cand["chg_pct"])
        c_vol = float(cand["volume"])
        c_vma20 = float(cand["vol_ma20_prev"]) if pd.notna(cand.get("vol_ma20_prev")) else float(cand.get("vol_ma20") or 0.0)
        v_mult = (c_vol / c_vma20) if c_vma20 > 0 else 0.0
        if cand["close"] > cand["open"]:
            if best_near_anchor is None or c_chg > best_near_anchor[0]:
                best_near_anchor = (c_chg, v_mult)
            if c_chg >= settings.track_b_anchor_min_chg_pct and v_mult >= settings.track_b_anchor_min_vol_ratio:
                anchor_idx = idx
                break

    if anchor_idx is None:
        hint = (
            f"(최근 최대 양봉 +{best_near_anchor[0]:.1f}%, 거래량 {best_near_anchor[1] * 100:.0f}%)"
            if best_near_anchor
            else "(4~15일 내 기준봉 없음)"
        )
        return {
            "track": "TRACK_B_PULLBACK",
            "track_label": "Track B (기준봉 눌림목 재점화)",
            "passed": False,
            "passed_count": 0,
            "total_checks": total_checks,
            "failed_reasons": [f"기준봉 조건 미충족 {hint}"],
            "invalidation_raw": None,
            "metrics": {},
        }

    anchor = df.iloc[anchor_idx]
    anchor_mid = (float(anchor["open"]) + float(anchor["close"])) / 2.0
    anchor_vol = float(anchor["volume"])
    anchor_obv = float(anchor["obv"])
    anchor_prev_obv = float(df["obv"].iloc[anchor_idx - 1]) if anchor_idx > 0 else anchor_obv

    adj_df = df.iloc[anchor_idx + 1 : last_idx]
    failed: list[str] = []

    # 2. 건전한 조정 (조정 기간 종가가 기준봉 중심값 및 20MA 미이탈)
    min_adj_close = float(adj_df["close"].min())
    adj_low_min = float(adj_df["low"].min())
    ma20_ok = bool((adj_df["close"] >= adj_df["ma20"]).all())
    mid_ok = min_adj_close >= anchor_mid
    if not (ma20_ok and mid_ok):
        adj_fails = []
        if not mid_ok:
            adj_fails.append(f"종가 기준봉 중심({anchor_mid:,.0f}) 이탈")
        if not ma20_ok:
            adj_fails.append("조정 중 20MA 이탈")
        failed.append(" / ".join(adj_fails))

    # 3. 거래량 건조 (조정 기간 평균 거래량 <= 기준봉 거래량의 45%)
    adj_vol_mean = float(adj_df["volume"].mean())
    adj_vol_ratio = (adj_vol_mean / anchor_vol) if anchor_vol > 0 else 9.9
    if adj_vol_ratio > settings.track_b_adj_max_vol_vs_anchor:
        failed.append(
            f"조정 거래량 건조 미흡 ({adj_vol_ratio * 100:.0f}% > {settings.track_b_adj_max_vol_vs_anchor * 100:.0f}%)"
        )

    # 4. 저위험 재점화 (5MA 또는 하락추세선 돌파 + 전일 대비 거래량 150% 이상)
    trend_ok = (ma5_today > 0 and close_today >= ma5_today) or _desc_trendline_broken(adj_df, close_today)
    vol_reignite_ratio = (vol_today / prev_vol) if prev_vol > 0 else 0.0
    if not (trend_ok and vol_reignite_ratio >= settings.track_b_reignite_min_vol_vs_prev):
        re_fails = []
        if not trend_ok:
            re_fails.append("5MA/추세선 재돌파 미완")
        if vol_reignite_ratio < settings.track_b_reignite_min_vol_vs_prev:
            re_fails.append(f"전일비 거래량 {vol_reignite_ratio * 100:.0f}% < 150%")
        failed.append(" / ".join(re_fails))

    # 5. 조정 중 OBV 훼손 없음 + 최근 3일 외인·기관 순매수 전환
    obv_intact = float(adj_df["obv"].min()) >= anchor_prev_obv and obv_today >= anchor_prev_obv
    frgn_3d = int(supply.get("frgn_3d", 0))
    orgn_3d = int(supply.get("orgn_3d", 0))
    frgn_today = int(supply.get("frgn_today", 0))
    orgn_today = int(supply.get("orgn_today", 0))
    supply_ok = (frgn_3d + orgn_3d > 0) or (frgn_today + orgn_today > 0)
    if not (obv_intact and supply_ok):
        s_fails = []
        if not obv_intact:
            s_fails.append("조정 중 OBV 훼손")
        if not supply_ok:
            s_fails.append("3일 수급 순매수 전환 미확인")
        failed.append(" & ".join(s_fails))

    invalidation_raw = min(adj_low_min, anchor_mid)
    return {
        "track": "TRACK_B_PULLBACK",
        "track_label": "Track B (기준봉 눌림목 재점화)",
        "passed": len(failed) == 0,
        "passed_count": total_checks - len(failed),
        "total_checks": total_checks,
        "failed_reasons": failed,
        "invalidation_raw": round(invalidation_raw, 2),
        "metrics": {
            "anchor_idx": anchor_idx,
            "anchor_date": str(anchor["date"]),
            "anchor_days_ago": last_idx - anchor_idx,
            "anchor_chg_pct": round(float(anchor["chg_pct"]), 2),
            "anchor_mid": round(anchor_mid, 2),
            "adj_vol_vs_anchor_pct": round(adj_vol_ratio * 100.0, 1),
            "reignite_vol_vs_prev_pct": round(vol_reignite_ratio * 100.0, 1),
        },
    }
