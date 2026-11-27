from __future__ import annotations

import logging
from typing import Any

from config.settings import settings
from indicators.technicals import compute_indicators
from providers.base import MarketDataProvider
from storage.repo import StorageRepo, repo as default_repo

logger = logging.getLogger(__name__)


def _diagnose_error_tags(
    signal: dict[str, Any],
    post_df,
    index_df,
    provider: MarketDataProvider,
) -> list[str]:
    """D+10 종료 시 목표 미달(+10% 또는 3R 실패) + 손실 마감 종목의 실패 원인 자동 태깅."""
    tags: list[str] = []
    if post_df is None or post_df.empty:
        return tags

    entry_p = float(signal.get("entry_price") or 0.0)
    rr = signal.get("rr_geometry") or {}
    inval_p = float(rr.get("invalidation_ref_price") or 0.0)
    nearest_res = rr.get("nearest_resistance_price")
    snap = signal.get("snapshot") or {}

    early = post_df.iloc[:2]

    # 1. RISK_BREACHED_EARLY: D+1~D+2 내 무효화 기준선 하향 이탈
    if inval_p > 0 and float(early["low"].min()) <= inval_p:
        tags.append("RISK_BREACHED_EARLY")

    # 2. SUPPLY_REVERSAL: D+1~D+2 외인·기관 합산 순매도 반전
    code = str(signal.get("code") or "").zfill(6)
    if len(early) >= 1:
        try:
            d1_d2_net = 0
            any_supply_row = False
            for _, erow in early.iterrows():
                d_early = str(erow["date"])
                sup = provider.get_investor_supply(code, days=1, end_date=d_early)
                if not sup.get("intraday_pending"):
                    d1_d2_net += int(sup.get("frgn_today", 0)) + int(sup.get("orgn_today", 0))
                    any_supply_row = True
            if any_supply_row and d1_d2_net < 0:
                tags.append("SUPPLY_REVERSAL")
        except Exception as e:
            logger.debug("error_log supply lookup failed (%s): %s", code, e)

    # 3. FAKEOUT_WICK: 돌파 당일 윗꼬리 과다(>=0.30) 또는 D+1 종가 진입가 이탈
    d1_close = float(post_df.iloc[0]["close"])
    if float(snap.get("upper_wick_ratio", 0.0)) >= 0.30 or (entry_p > 0 and d1_close < entry_p):
        tags.append("FAKEOUT_WICK")

    # 4. INDEX_DRAG: D+1~D+2 KOSPI 지수 -1.5% 이상 동반 급락
    if index_df is not None and not index_df.empty:
        early_dates = set(early["date"].astype(str).tolist())
        idx_early = index_df[index_df["date"].isin(early_dates)]
        if not idx_early.empty and float(idx_early["chg_pct"].min()) <= settings.weak_regime_index_drop_pct:
            tags.append("INDEX_DRAG")

    # 5. RESISTANCE_REJECT: 직전 매물대 부근(2% 이내) 접근 후 거래량 동반 음봉 꺾임
    if nearest_res is not None and float(nearest_res) > 0:
        res_p = float(nearest_res)
        near_bars = post_df[post_df["high"] >= res_p * 0.98]
        if not near_bars.empty:
            for _, bar in near_bars.iterrows():
                vma = float(bar.get("vol_ma20_prev") or bar.get("vol_ma20") or 0.0)
                if float(bar["close"]) < float(bar["open"]) and (vma <= 0 or float(bar["volume"]) >= vma):
                    tags.append("RESISTANCE_REJECT")
                    break

    return list(dict.fromkeys(tags))


def update_error_logs(
    provider: MarketDataProvider,
    storage: StorageRepo | None = None,
    as_of_date: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """진행 중(ACTIVE) 시그널의 D+1~D+10 MFE/MAE·R배수 및 완료 오답 태그를 갱신."""
    store = storage or default_repo
    active_signals = store.get_active_signals()
    if not active_signals:
        return {"active_checked": 0, "updated": 0, "completed": 0, "details": []}

    index_df = provider.get_index_ohlcv(bars=60, end_date=as_of_date)
    updated_count = 0
    completed_count = 0
    details: list[dict[str, Any]] = []

    for sig in active_signals:
        code = str(sig.get("code") or "").zfill(6)
        signal_date = str(sig.get("signal_date") or "")
        entry_p = float(sig.get("entry_price") or 0.0)
        rr = sig.get("rr_geometry") or {}
        risk_pct = float(rr.get("risk_pct") or 0.0)
        if not code or not signal_date or entry_p <= 0 or risk_pct <= 0:
            continue

        raw_df = provider.get_daily_ohlcv(code, bars=60, end_date=as_of_date)
        df_ind = compute_indicators(raw_df)
        if df_ind.empty:
            continue

        post_df = df_ind[df_ind["date"] > signal_date].sort_values("date").reset_index(drop=True)
        post_10 = post_df.head(settings.error_tracking_max_days)
        if post_10.empty:
            continue

        days_elapsed = len(post_10)
        mfe_pct = round(((float(post_10["high"].max()) - entry_p) / entry_p) * 100.0, 2)
        mae_pct = round(((float(post_10["low"].min()) - entry_p) / entry_p) * 100.0, 2)
        max_r = round(max(0.0, mfe_pct) / risk_pct, 2)

        hit_10 = bool(mfe_pct >= 10.0)
        hit_20 = bool(mfe_pct >= 20.0)
        hit_30 = bool(mfe_pct >= 30.0)
        hit_3r = bool(max_r >= settings.min_projected_rr)

        closes_map: dict[str, int | None] = {}
        for d_n in (1, 3, 5, 7, 10):
            key = f"d{d_n}"
            if days_elapsed >= d_n:
                closes_map[key] = int(round(float(post_10.iloc[d_n - 1]["close"])))
            else:
                closes_map[key] = None

        is_completed = days_elapsed >= settings.error_tracking_max_days
        status = "COMPLETED" if is_completed else "ACTIVE"

        error_tags = list(sig.get("error_tags") or [])
        last_close = float(post_10.iloc[-1]["close"])
        if is_completed and (not hit_10 or not hit_3r) and last_close < entry_p:
            error_tags = _diagnose_error_tags(sig, post_10, index_df, provider)

        updated_sig = dict(sig)
        updated_sig["tracking"] = {
            "status": status,
            "days_elapsed": days_elapsed,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "max_r_multiple": max_r,
            "hit_10pct": hit_10,
            "hit_20pct": hit_20,
            "hit_30pct": hit_30,
            "hit_3r": hit_3r,
            "closes": closes_map,
        }
        updated_sig["error_tags"] = error_tags

        if not dry_run:
            store.upsert_signal(updated_sig)

        updated_count += 1
        if is_completed:
            completed_count += 1
        details.append(
            {
                "signal_id": sig.get("signal_id"),
                "code": code,
                "name": sig.get("name"),
                "status": status,
                "days_elapsed": days_elapsed,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "max_r_multiple": max_r,
                "error_tags": error_tags,
            }
        )

    return {
        "active_checked": len(active_signals),
        "updated": updated_count,
        "completed": completed_count,
        "details": details,
    }
