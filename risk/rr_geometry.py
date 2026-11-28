from __future__ import annotations

from typing import Any

import pandas as pd

from config.settings import settings
from risk.krx_tick import round_to_tick


def _max_risk_limit_for_track(track: str) -> float:
    t = str(track or "").upper()
    if "TRACK_A" in t:
        return settings.track_a_max_risk_pct
    if "TRACK_B" in t:
        return settings.track_b_max_risk_pct
    if "TRACK_C" in t:
        return settings.track_c_max_risk_pct
    if "TRACK_D" in t:
        return settings.track_a_max_risk_pct
    return settings.track_c_max_risk_pct


def evaluate_rr_geometry(
    df: pd.DataFrame,
    entry_price: float,
    invalidation_raw: float,
    track: str,
    impulse_start_idx: int | None = None,
    min_rr_override: float | None = None,
    risk_mult: float = 1.0,
) -> dict[str, Any]:
    """1:3 손익비 성립 여부를 기하학(구조적 리스크 + 상단 매물대 공백)으로 사전 검증한다."""
    max_risk_pct = round(_max_risk_limit_for_track(track) * float(risk_mult), 2)
    eff_min_rr = float(min_rr_override) if min_rr_override is not None else settings.min_projected_rr
    entry_tick = round_to_tick(entry_price, mode="nearest")
    inval_tick = round_to_tick(invalidation_raw, mode="down")

    base_resp: dict[str, Any] = {
        "passed": False,
        "reject_reason": "",
        "failed_reasons": [],
        "entry_price": entry_tick,
        "max_entry_price": entry_tick,
        "invalidation_ref_price": inval_tick,
        "risk_pct": 0.0,
        "max_risk_pct": max_risk_pct,
        "nearest_resistance_price": None,
        "clear_upside_pct": 0.0,
        "projected_rr": 0.0,
        "is_blue_sky": False,
    }

    if df is None or len(df) < 20 or entry_tick <= 0:
        base_resp["reject_reason"] = "기초 일봉 데이터 부족"
        base_resp["failed_reasons"] = ["기초 일봉 데이터 부족"]
        return base_resp

    if inval_tick <= 0 or inval_tick >= entry_tick:
        msg = f"무효화 기준선 오류 ({inval_tick:,}원 >= 진입가 {entry_tick:,}원)"
        base_resp["reject_reason"] = msg
        base_resp["failed_reasons"] = [msg]
        return base_resp

    risk_pct = round(((entry_tick - inval_tick) / entry_tick) * 100.0, 2)
    base_resp["risk_pct"] = risk_pct

    if risk_pct < 0.5:
        msg = f"무효화 폭 극소 ({risk_pct:.2f}% < 0.50% 노이즈 구간)"
        base_resp["reject_reason"] = msg
        base_resp["failed_reasons"] = [msg]
        return base_resp

    # 최근 120거래일(당일 제외) 중 거래량이 직전 20일 평균의 2배 이상 터진 매물대 저항 탐색
    # 단, Track B(기준봉 눌림목)는 기준봉 자체 고점을 과거 매물대로 오인하지 않도록 기준봉 이전 구간에서 탐색
    if impulse_start_idx is not None and 20 <= impulse_start_idx < len(df):
        hist = df.iloc[:impulse_start_idx].tail(settings.resistance_lookback_days).copy()
    elif "TRACK_B" in str(track).upper() and len(df) > (settings.track_b_lookback_max_days + 25):
        hist = df.iloc[: -(settings.track_b_lookback_max_days + 1)].tail(settings.resistance_lookback_days).copy()
    else:
        hist = df.iloc[:-1].tail(settings.resistance_lookback_days).copy()
    if "vol_ma20_prev" in hist.columns:
        vol_base = hist["vol_ma20_prev"].fillna(hist.get("vol_ma20", 0.0))
    elif "vol_ma20" in hist.columns:
        vol_base = hist["vol_ma20"]
    else:
        vol_base = hist["volume"].rolling(20, min_periods=5).mean().shift(1)

    vol_ratio_ok = (vol_base > 0) & (hist["volume"] >= settings.heavy_vol_resistance_mult * vol_base)
    above_entry = hist["high"] > (entry_tick * 1.005)
    heavy_candidates = hist[vol_ratio_ok & above_entry]

    prior_120_high = float(hist["high"].max()) if not hist.empty else 0.0
    is_near_or_new_120h = prior_120_high <= 0 or entry_tick >= (prior_120_high * 0.99)

    nearest_res: int | None = None
    is_blue_sky = False
    if not heavy_candidates.empty:
        raw_nearest = float(heavy_candidates["high"].min())
        nearest_res = round_to_tick(raw_nearest, mode="up")
        clear_upside_pct = round(((nearest_res - entry_tick) / entry_tick) * 100.0, 2)
    elif is_near_or_new_120h:
        is_blue_sky = True
        clear_upside_pct = float(settings.blue_sky_upside_pct)
    else:
        # 120일 신고가 돌파가 아닌데 2배 거래량 매물대만 없는 경우 보수적으로 120일 전고점까지를 공간으로 측정
        nearest_res = round_to_tick(prior_120_high, mode="up")
        clear_upside_pct = round(((nearest_res - entry_tick) / entry_tick) * 100.0, 2)

    projected_rr = round(clear_upside_pct / risk_pct, 2) if risk_pct > 0 else 0.0

    # 정규장 기준 1:3 R:R 및 최대 허용 리스크가 유지되는 추격 허용 상한가 계산
    target_ceiling = (
        float(nearest_res)
        if nearest_res is not None
        else float(entry_tick) * (1.0 + clear_upside_pct / 100.0)
    )
    max_by_risk = float(inval_tick) / max(0.01, 1.0 - (max_risk_pct / 100.0))
    max_by_rr = (target_ceiling + (eff_min_rr * float(inval_tick))) / (1.0 + eff_min_rr)
    max_by_upside = target_ceiling / (1.0 + settings.min_clear_upside_pct / 100.0)
    max_by_chase = float(entry_tick) * 1.03
    max_entry_price = max(
        entry_tick,
        round_to_tick(min(max_by_risk, max_by_rr, max_by_upside, max_by_chase), mode="down"),
    )

    failed: list[str] = []
    if risk_pct > max_risk_pct:
        failed.append(f"진입 지연(손익비 불량: 리스크 {risk_pct:.2f}% > 상한 {max_risk_pct:.1f}%)")
    if clear_upside_pct < settings.min_clear_upside_pct:
        failed.append(
            f"상단 매물대 공백 부족 ({clear_upside_pct:.2f}% < {settings.min_clear_upside_pct:.1f}%)"
        )
    if projected_rr < eff_min_rr:
        failed.append(f"예상 손익비 미달 (1:{projected_rr:.2f} < 1:{eff_min_rr:.1f})")

    base_resp.update(
        {
            "passed": len(failed) == 0,
            "reject_reason": failed[0] if failed else "",
            "failed_reasons": failed,
            "max_entry_price": max_entry_price,
            "nearest_resistance_price": nearest_res,
            "clear_upside_pct": clear_upside_pct,
            "projected_rr": projected_rr,
            "is_blue_sky": is_blue_sky,
        }
    )
    return base_resp
