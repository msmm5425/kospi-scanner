from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from config.settings import KST, settings
from indicators.technicals import compute_bb_squeeze_percentile, compute_indicators
from providers.base import MarketDataProvider, get_market_index_ohlcv
from risk.rr_geometry import evaluate_rr_geometry
from storage.repo import StorageRepo, repo as default_repo
from strategies.common_gate import evaluate_common_gate
from strategies.track_a_vcp import evaluate_track_a
from strategies.track_b_pullback import evaluate_track_b
from strategies.track_c_turnaround import evaluate_track_c

logger = logging.getLogger(__name__)


def _diagnose_why_missed(
    code: str,
    df_full: pd.DataFrame,
    index_df: pd.DataFrame,
    marcap_krw: float,
    provider: MarketDataProvider,
    market: str = "KOSPI",
) -> dict[str, Any]:
    """D-1 ~ D-3 시점 데이터로 역산해 가장 적은 미달 항목으로 탈락한 병목 원인 반환."""
    best_candidate: dict[str, Any] | None = None

    for d_offset in (1, 2, 3):
        if len(df_full) <= (d_offset + 25):
            continue
        sub_raw = df_full.iloc[:-d_offset].copy().reset_index(drop=True)
        sub_ind = compute_indicators(sub_raw)
        if sub_ind.empty or len(sub_ind) < 25:
            continue

        snap_date = str(sub_ind.iloc[-1]["date"])
        idx_slice = (
            index_df[index_df["date"] <= snap_date].reset_index(drop=True)
            if not index_df.empty
            else index_df
        )
        tval_hist = float(sub_ind.iloc[-1].get("amount") or 0.0)
        gate_res = evaluate_common_gate(
            df=sub_ind,
            index_df=idx_slice,
            trading_val=tval_hist,
            marcap_krw=marcap_krw,
            min_trading_val_override=settings.top20_min_trading_val_krw,
            market=market,
        )
        supply = provider.get_investor_supply(code, days=settings.supply_days, end_date=snap_date)

        for evaluator in (evaluate_track_a, evaluate_track_b, evaluate_track_c):
            tr = evaluator(sub_ind, supply)
            failures: list[str] = []
            if not gate_res["passed"]:
                failures.extend(gate_res["failed_reasons"])
            failures.extend(tr["failed_reasons"])

            # 전략 통과 시 R:R 병목 확인
            if tr["passed"] and tr.get("invalidation_raw") is not None:
                rr = evaluate_rr_geometry(
                    df=sub_ind,
                    entry_price=float(sub_ind.iloc[-1]["close"]),
                    invalidation_raw=float(tr["invalidation_raw"]),
                    track=tr["track"],
                    impulse_start_idx=(tr.get("metrics") or {}).get("anchor_idx"),
                )
                if not rr["passed"]:
                    failures.extend(rr["failed_reasons"])

            total_items = 7 + tr["total_checks"]
            fail_cnt = max(1, len(failures))
            score = (total_items - fail_cnt, -d_offset)
            summary = (
                f"D-{d_offset} {tr['track_label'].split('(')[0].strip()} 근접 탈락 "
                f"[{'; '.join(failures[:2])}]"
            )

            cand = {
                "score": score,
                "day_offset": f"D-{d_offset}",
                "nearest_track": tr["track"],
                "failure_count": fail_cnt,
                "failed_reasons": failures,
                "summary": summary,
            }
            if best_candidate is None or cand["score"] > best_candidate["score"]:
                best_candidate = cand

    if best_candidate is None:
        return {
            "day_offset": "N/A",
            "nearest_track": "NONE",
            "failure_count": 99,
            "failed_reasons": ["과거 일봉 데이터 부족"],
            "summary": "과거 일봉 부족으로 역추적 불가",
        }
    best_candidate.pop("score", None)
    return best_candidate


def _extract_standard_features(
    code: str,
    df_full: pd.DataFrame,
    supply_d1: dict[str, Any],
    sector_sync_codes: set[str],
) -> tuple[list[str], dict[str, Any]]:
    """Track D 발굴용 6개 표준 전조 피처 태깅 + 급등 전 1:3 R:R 가능성 평가."""
    tags: list[str] = []
    if df_full is None or len(df_full) < 25:
        return tags, {"rr_viable_1to3": False, "pre_entry_rr": 0.0}

    df_ind = compute_indicators(df_full)
    pre_df = df_ind.iloc[:-1].copy().reset_index(drop=True)
    today = df_ind.iloc[-1]
    prev = pre_df.iloc[-1]

    # 1. BB_EXTREME_SQUEEZE (D-1 기준 60일 하위 10%)
    bb_pct = compute_bb_squeeze_percentile(pre_df, window=60)
    if bb_pct is not None and bb_pct <= 10.0:
        tags.append("BB_EXTREME_SQUEEZE")

    # 2. ORG_STEALTH_5D (주가 횡보/소폭 조정 중 기관 5일 연속 순매수)
    orgn_series = supply_d1.get("orgn_series") or []
    if len(pre_df) >= 6:
        base_c = float(pre_df.iloc[-6]["close"])
        end_c = float(prev["close"])
        ret_5d = ((end_c - base_c) / base_c * 100.0) if base_c > 0 else 99.0
        if len(orgn_series) >= 5 and all(x > 0 for x in orgn_series[-5:]) and (-6.0 <= ret_5d <= 6.0):
            tags.append("ORG_STEALTH_5D")

    # 3. MA_CONVERGENCE (D-1 시점 5/20/60일선 3% 이내 밀집 후 당일 발산)
    ma5 = float(prev["ma5"]) if pd.notna(prev.get("ma5")) else 0.0
    ma20 = float(prev["ma20"]) if pd.notna(prev.get("ma20")) else 0.0
    ma60 = float(prev["ma60"]) if pd.notna(prev.get("ma60")) else 0.0
    if ma5 > 0 and ma20 > 0 and ma60 > 0:
        spread = (max(ma5, ma20, ma60) / min(ma5, ma20, ma60)) - 1.0
        t_ma5 = float(today["ma5"]) if pd.notna(today.get("ma5")) else 0.0
        t_ma20 = float(today["ma20"]) if pd.notna(today.get("ma20")) else 0.0
        if spread <= 0.03 and float(today["close"]) > float(today["open"]) and t_ma5 >= t_ma20:
            tags.append("MA_CONVERGENCE")

    # 4. GAP_BREAKAWAY (당일 +3% 이상 갭상승 + 양봉 + 거래량 2배)
    prev_c = float(prev["close"])
    gap_pct = ((float(today["open"]) - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
    v_ma20_p = float(today.get("vol_ma20_prev") or 0.0)
    if gap_pct >= 3.0 and float(today["close"]) >= float(today["open"]) and v_ma20_p > 0 and float(today["volume"]) >= v_ma20_p * 2.0:
        tags.append("GAP_BREAKAWAY")

    # 5. UPPER_SHADOW_ABSORB (직전 10일 내 대량 거래 윗꼬리 매집봉 2회 이상)
    pre_10 = pre_df.tail(10)
    absorb_cnt = 0
    for _, bar in pre_10.iterrows():
        vma = float(bar.get("vol_ma20_prev") or bar.get("vol_ma20") or 0.0)
        if float(bar.get("upper_wick_ratio", 0.0)) >= 0.35 and vma > 0 and float(bar["volume"]) >= vma * 1.5:
            absorb_cnt += 1
    if absorb_cnt >= 2:
        tags.append("UPPER_SHADOW_ABSORB")

    # 6. SECTOR_SYNC (동일 업종 내 3개 이상 동반 강세)
    if code in sector_sync_codes:
        tags.append("SECTOR_SYNC")

    # D-1 시점 진입 가정 시 1:3 R:R 가능 여부 체크
    pre_low_5 = float(pre_df.tail(5)["low"].min())
    rr_check = evaluate_rr_geometry(
        df=pre_df,
        entry_price=float(prev["close"]),
        invalidation_raw=pre_low_5,
        track="TRACK_A_VCP",
    )
    return tags, {
        "rr_viable_1to3": bool(rr_check["passed"]),
        "pre_entry_rr": rr_check["projected_rr"],
    }


def analyze_top20(
    provider: MarketDataProvider,
    storage: StorageRepo | None = None,
    as_of_date: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """15:50 코스피·코스닥 상승률 TOP 20 역추적(Hit vs Miss) 및 Track D 후보 패턴 큐 누적."""
    store = storage or default_repo
    index_df = get_market_index_ohlcv(provider, bars=100, end_date=as_of_date, market="KOSPI")
    trading_dates = index_df["date"].astype(str).tolist() if not index_df.empty else []
    target_date = (
        trading_dates[-1]
        if trading_dates
        else (as_of_date or datetime.now(tz=KST).strftime("%Y-%m-%d"))
    )

    top20 = provider.get_top_gainers(top_n=20, end_date=as_of_date)
    index_by_market = {"KOSPI": index_df}
    if any(str(g.get("market") or "KOSPI").upper() == "KOSDAQ" for g in top20):
        index_by_market["KOSDAQ"] = get_market_index_ohlcv(provider, bars=100, end_date=as_of_date, market="KOSDAQ")
    recent_signals = store.get_recent_signals_by_trading_days(
        target_date=target_date,
        trading_dates=trading_dates or [target_date],
        lookback_days=settings.top20_lookback_trading_days,
    )
    recent_by_code = {str(s["code"]).zfill(6): s for s in recent_signals}

    # SECTOR_SYNC 탐지: 동일 업종 3종목 이상 TOP 20 동반 포진
    sector_map = provider.get_sector_map()
    sector_counts: dict[str, list[str]] = {}
    for g in top20:
        c = str(g["code"]).zfill(6)
        sec = sector_map.get(c, "")
        if sec:
            sector_counts.setdefault(sec, []).append(c)
    sector_sync_codes: set[str] = set()
    for codes in sector_counts.values():
        if len(codes) >= 3:
            sector_sync_codes.update(codes)

    hits: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    pattern_entries: list[dict[str, Any]] = []

    for g in top20:
        code = str(g["code"]).zfill(6)
        name = str(g["name"]).strip()
        mkt = str(g.get("market") or "KOSPI").upper()
        chg_pct = float(g.get("chg_pct", 0.0))
        marcap_krw = float(g.get("marcap", 0.0))

        if code in recent_by_code:
            sig = recent_by_code[code]
            hits.append(
                {
                    "code": code,
                    "name": name,
                    "market": mkt,
                    "chg_pct": chg_pct,
                    "signal_id": sig["signal_id"],
                    "signal_date": sig["signal_date"],
                    "track": sig["track"],
                }
            )
            continue

        mkt_idx_df = index_by_market.get(mkt, index_df)
        df_full = provider.get_daily_ohlcv(code, bars=settings.ohlcv_bars + 5, end_date=target_date)
        diag = _diagnose_why_missed(code, df_full, mkt_idx_df, marcap_krw, provider, market=mkt)
        d1_date = str(df_full.iloc[-2]["date"]) if len(df_full) >= 2 else target_date
        supply_d1 = provider.get_investor_supply(code, days=5, end_date=d1_date)
        feat_tags, rr_meta = _extract_standard_features(code, df_full, supply_d1, sector_sync_codes)

        miss_rec = {
            "code": code,
            "name": name,
            "market": mkt,
            "chg_pct": chg_pct,
            "trading_val_uk": round(float(g.get("trading_val", 0.0)) / 1e8, 1),
            "diagnosis": diag["summary"],
            "nearest_track": diag["nearest_track"],
            "feature_tags": feat_tags,
            "rr_viable_1to3": rr_meta["rr_viable_1to3"],
        }
        misses.append(miss_rec)

        if feat_tags:
            pattern_entries.append(
                {
                    "date": target_date,
                    "code": code,
                    "name": name,
                    "market": mkt,
                    "chg_pct": chg_pct,
                    "nearest_track": diag["nearest_track"],
                    "tags": feat_tags,
                    "rr_viable_1to3": rr_meta["rr_viable_1to3"],
                    "pre_entry_rr": rr_meta["pre_entry_rr"],
                }
            )

    hit_rate = round((len(hits) / len(top20)) * 100.0, 1) if top20 else 0.0
    day_summary = {
        "date": target_date,
        "top20_count": len(top20),
        "hit_count": len(hits),
        "miss_count": len(misses),
        "hit_rate_pct": hit_rate,
        "hits": hits,
        "misses": misses,
    }

    if not dry_run:
        store.upsert_top20_day(day_summary)
        store.append_pattern_entries(pattern_entries)

    return {
        "summary": day_summary,
        "pattern_entries_added": len(pattern_entries),
    }
