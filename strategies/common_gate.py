from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from config.settings import KST, get_dynamic_min_trading_val_krw, settings
from indicators.technicals import compute_relative_strength


def evaluate_market_regime(index_df: pd.DataFrame, market: str = "KOSPI") -> dict[str, Any]:
    """KOSPI/KOSDAQ 지수 20일선 하회 또는 당일 -1.5% 이하 급락 시 WEAK 태깅."""
    mkt_label = "KOSDAQ" if str(market or "KOSPI").upper() == "KOSDAQ" else "KOSPI"
    if index_df is None or index_df.empty:
        return {
            "market_regime": "NORMAL",
            "kospi_close": 0.0,
            "kospi_ma20": 0.0,
            "kospi_chg_pct": 0.0,
            "reason": "지수 데이터 없음(기본 NORMAL)",
        }

    closes = pd.to_numeric(index_df["close"], errors="coerce").dropna()
    chgs = pd.to_numeric(index_df["chg_pct"], errors="coerce").fillna(0.0)
    last_close = float(closes.iloc[-1]) if not closes.empty else 0.0
    last_chg = float(chgs.iloc[-1]) if not chgs.empty else 0.0
    ma20 = float(closes.tail(20).mean()) if len(closes) >= 20 else float(closes.mean())

    reasons: list[str] = []
    if ma20 > 0 and last_close < ma20:
        reasons.append(f"{mkt_label} 20일선 하회({last_close:,.2f} < {ma20:,.2f})")
    if last_chg <= settings.weak_regime_index_drop_pct:
        reasons.append(f"{mkt_label} 급락({last_chg:+.2f}%)")

    regime = "WEAK" if reasons else "NORMAL"
    return {
        "market_regime": regime,
        "kospi_close": round(last_close, 2),
        "kospi_ma20": round(ma20, 2),
        "kospi_chg_pct": round(last_chg, 2),
        "reason": " / ".join(reasons) if reasons else "정상/강세",
    }


def evaluate_common_gate(
    df: pd.DataFrame,
    index_df: pd.DataFrame,
    trading_val: float,
    marcap_krw: float,
    now_kst: datetime | None = None,
    min_trading_val_override: int | None = None,
    max_upper_wick_override: float | None = None,
    max_disparity_override: float | None = None,
    market: str = "KOSPI",
) -> dict[str, Any]:
    """Track A/B/C 검사 전 공통 기술적 · 추격 방지 · 유동성 관문 평가."""
    regime_info = evaluate_market_regime(index_df, market=market)
    min_tval = (
        int(min_trading_val_override)
        if min_trading_val_override is not None
        else get_dynamic_min_trading_val_krw(now_kst or datetime.now(tz=KST), market=market)
    )
    max_wick = (
        float(max_upper_wick_override)
        if max_upper_wick_override is not None
        else settings.max_upper_wick_ratio
    )
    max_disp = (
        float(max_disparity_override)
        if max_disparity_override is not None
        else settings.max_disparity_20_pct
    )

    if df is None or len(df) < 25:
        return {
            "passed": False,
            "hard_failed": True,
            "failed_reasons": ["일봉 데이터 부족(25봉 미만)"],
            "market_regime": regime_info["market_regime"],
            "regime_info": regime_info,
            "metrics": {},
        }

    row = df.iloc[-1]
    open_p = float(row["open"])
    close_p = float(row["close"])
    chg_pct = float(row.get("chg_pct", 0.0))
    wick_ratio = float(row.get("upper_wick_ratio", 0.0))
    disp_20 = float(row.get("disparity_20", 0.0)) if pd.notna(row.get("disparity_20")) else 0.0
    eff_tval = float(trading_val if trading_val > 0 else row.get("amount", 0.0))

    rs_info = compute_relative_strength(df, index_df)
    rs_20 = float(rs_info["rs_20d"])
    rs_60 = float(rs_info["rs_60d"])

    failed: list[str] = []
    if eff_tval < min_tval:
        failed.append(f"거래대금 부족 ({eff_tval / 1e8:.1f}억 < {min_tval / 1e8:.0f}억)")
    if close_p <= open_p:
        failed.append(f"당일 음봉/도지 (종가 {close_p:,.0f} <= 시가 {open_p:,.0f})")
    if chg_pct < settings.min_chg_pct:
        failed.append(f"당일 상승 탄력 부족 ({chg_pct:+.2f}% < +{settings.min_chg_pct:.1f}%)")
    elif chg_pct > settings.max_chg_pct:
        failed.append(f"당일 급등 추격 차단 ({chg_pct:+.2f}% > +{settings.max_chg_pct:.1f}%)")
    if wick_ratio > max_wick:
        failed.append(f"윗꼬리 과다 ({wick_ratio:.2f} > {max_wick:.2f})")
    if disp_20 <= 0 or disp_20 > max_disp:
        failed.append(f"20일선 이격 과열 ({disp_20:.1f}% > {max_disp:.0f}%)")
    if rs_20 <= 0.0 and rs_60 <= 0.0:
        failed.append(f"상대강도 열위 (RS20 {rs_20:+.1f}%, RS60 {rs_60:+.1f}%)")

    vol_ma20_ref = float(row["vol_ma20_prev"]) if pd.notna(row.get("vol_ma20_prev")) and row.get("vol_ma20_prev", 0) > 0 else float(row.get("vol_ma20") or 0.0)
    vol_ratio_20d = round((float(row["volume"]) / vol_ma20_ref) * 100.0, 1) if vol_ma20_ref > 0 else 0.0
    atr14_pct = round(float(row["atr14_pct"]), 2) if pd.notna(row.get("atr14_pct")) else 0.0

    return {
        "passed": len(failed) == 0,
        "failed_reasons": failed,
        "market_regime": regime_info["market_regime"],
        "regime_info": regime_info,
        "metrics": {
            "close": close_p,
            "chg_pct": round(chg_pct, 2),
            "marcap_uk": round(float(marcap_krw) / 1e8, 1),
            "trading_val_uk": round(eff_tval / 1e8, 1),
            "upper_wick_ratio": round(wick_ratio, 3),
            "disparity_20": round(disp_20, 2),
            "atr14_pct": atr14_pct,
            "vol_ratio_20d": vol_ratio_20d,
            "rs_20d_diff": rs_20,
            "rs_60d_diff": rs_60,
        },
    }
