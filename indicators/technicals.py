from __future__ import annotations

import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """일봉 OHLCV에 MA, ATR(14), OBV, 볼린저밴드, 변동폭, 이격도 지표를 미래 참조 없이 계산한다."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy().sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume", "amount"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)

    prev_close = out["close"].shift(1)
    out["chg_pct"] = ((out["close"] - prev_close) / prev_close.replace(0.0, np.nan)) * 100.0
    out["chg_pct"] = out["chg_pct"].fillna(0.0)

    for w in (5, 10, 20, 60, 120):
        out[f"ma{w}"] = out["close"].rolling(window=w, min_periods=w).mean()

    out["vol_ma5"] = out["volume"].rolling(window=5, min_periods=5).mean()
    out["vol_ma20"] = out["volume"].rolling(window=20, min_periods=20).mean()
    # 당일 거래량 급증이 평균을 왜곡하지 않도록 직전 20일 거래량 평균 별도 산출
    out["vol_ma20_prev"] = out["vol_ma20"].shift(1)

    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    out["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr14"] = out["tr"].rolling(window=14, min_periods=5).mean()
    out["atr14_pct"] = (out["atr14"] / out["close"].replace(0.0, np.nan)) * 100.0

    # OBV
    diff = out["close"].diff().fillna(0.0)
    direction = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
    out["obv"] = (out["volume"] * direction).cumsum()
    out["obv_ma5"] = out["obv"].rolling(window=5, min_periods=1).mean()

    # Bollinger Bands (20, 2)
    bb_mid = out["ma20"]
    bb_std = out["close"].rolling(window=20, min_periods=20).std(ddof=0)
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_mid + (2.0 * bb_std)
    out["bb_lower"] = bb_mid - (2.0 * bb_std)
    out["bb_width"] = ((out["bb_upper"] - out["bb_lower"]) / bb_mid.replace(0.0, np.nan)) * 100.0

    # 일중 변동폭(%), 윗꼬리 비율, 20일선 이격도(%)
    denom_p = prev_close.where(prev_close > 0, out["open"]).replace(0.0, np.nan)
    out["range_pct"] = ((out["high"] - out["low"]).clip(lower=0.0) / denom_p) * 100.0

    hl_span = (out["high"] - out["low"]).replace(0.0, np.nan)
    out["upper_wick_ratio"] = ((out["high"] - out["close"]) / hl_span).clip(lower=0.0, upper=1.0).fillna(0.0)
    out["disparity_20"] = (out["close"] / out["ma20"].replace(0.0, np.nan)) * 100.0

    return out


def _period_return_pct(series: pd.Series, lookback: int) -> float:
    if series is None or len(series) < 2:
        return 0.0
    valid = series.dropna()
    if len(valid) < 2:
        return 0.0
    start_idx = max(0, len(valid) - 1 - lookback)
    base = float(valid.iloc[start_idx])
    last = float(valid.iloc[-1])
    if base <= 0:
        return 0.0
    return ((last - base) / base) * 100.0


def compute_relative_strength(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> dict[str, float]:
    """KOSPI 지수 대비 20일·60일 초과 수익률(RS)을 날짜 정렬 기준으로 산출."""
    result = {
        "rs_20d": 0.0,
        "rs_60d": 0.0,
        "stock_ret_20d": 0.0,
        "kospi_ret_20d": 0.0,
        "stock_ret_60d": 0.0,
        "kospi_ret_60d": 0.0,
    }
    if stock_df is None or stock_df.empty or index_df is None or index_df.empty:
        return result

    s_merged = stock_df[["date", "close"]].merge(
        index_df[["date", "close"]],
        on="date",
        how="inner",
        suffixes=("_stk", "_idx"),
    )
    if len(s_merged) >= 5:
        s_20 = _period_return_pct(s_merged["close_stk"], 20)
        k_20 = _period_return_pct(s_merged["close_idx"], 20)
        s_60 = _period_return_pct(s_merged["close_stk"], 60)
        k_60 = _period_return_pct(s_merged["close_idx"], 60)
    else:
        s_20 = _period_return_pct(stock_df["close"], 20)
        k_20 = _period_return_pct(index_df["close"], 20)
        s_60 = _period_return_pct(stock_df["close"], 60)
        k_60 = _period_return_pct(index_df["close"], 60)

    result.update(
        {
            "rs_20d": round(s_20 - k_20, 2),
            "rs_60d": round(s_60 - k_60, 2),
            "stock_ret_20d": round(s_20, 2),
            "kospi_ret_20d": round(k_20, 2),
            "stock_ret_60d": round(s_60, 2),
            "kospi_ret_60d": round(k_60, 2),
        }
    )
    return result


def compute_vcp_contraction_ratio(df: pd.DataFrame) -> tuple[float, float, float]:
    """돌파 당일을 제외한 직전 5일 평균 일중 변동폭 vs 직전 20일 평균 일중 변동폭 비율 계산.

    Returns:
        (recent_5d_avg_pct, prior_20d_avg_pct, ratio)
    """
    if df is None or len(df) < 22 or "range_pct" not in df.columns:
        return (0.0, 0.0, 99.0)

    pre_5 = df["range_pct"].iloc[-6:-1].dropna()
    if len(df) >= 27:
        prior_20 = df["range_pct"].iloc[-26:-6].dropna()
    else:
        prior_20 = df["range_pct"].iloc[:-1].tail(20).dropna()

    if len(pre_5) < 5 or len(prior_20) < 10:
        return (0.0, 0.0, 99.0)

    avg_5 = float(pre_5.mean())
    avg_20 = float(prior_20.mean())
    if avg_20 <= 1e-6:
        return (round(avg_5, 3), 0.0, 99.0)
    return (round(avg_5, 3), round(avg_20, 3), round(avg_5 / avg_20, 3))


def compute_bb_squeeze_percentile(df: pd.DataFrame, window: int = 60) -> float | None:
    """직전(당일 제외) 또는 현재 볼린저밴드 폭의 최근 window일 내 백분위(0~100) 반환."""
    if df is None or "bb_width" not in df.columns:
        return None
    sub = df["bb_width"].dropna().tail(window)
    if len(sub) < 20:
        return None
    curr = float(sub.iloc[-1])
    pct_rank = float((sub <= curr).sum()) / float(len(sub)) * 100.0
    return round(pct_rank, 1)
