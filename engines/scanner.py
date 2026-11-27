from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from config.settings import KST, get_dynamic_min_trading_val_krw, settings
from indicators.technicals import compute_indicators
from notify.telegram_bot import TelegramNotifier
from providers.base import MarketDataProvider
from risk.rr_geometry import evaluate_rr_geometry
from storage.repo import StorageRepo, repo as default_repo
from strategies.common_gate import evaluate_common_gate
from strategies.track_a_vcp import evaluate_track_a
from strategies.track_b_pullback import evaluate_track_b
from strategies.track_c_turnaround import evaluate_track_c

logger = logging.getLogger(__name__)

_TRACK_SHORT_SUFFIX = {
    "TRACK_A_VCP": "A",
    "TRACK_B_PULLBACK": "B",
    "TRACK_C_TURNAROUND": "C",
}


def _evaluate_priority_tracks(
    df_ind,
    supply: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """우선순위 Track A -> B -> C 순서대로 평가해 첫 통과 트랙과 전체 진단 목록을 반환."""
    diagnostics: list[dict[str, Any]] = []
    for evaluator in (evaluate_track_a, evaluate_track_b, evaluate_track_c):
        res = evaluator(df_ind, supply)
        diagnostics.append(res)
        if res["passed"]:
            return res, diagnostics
    return None, diagnostics


def _build_signal_record(
    signal_date: str,
    code: str,
    name: str,
    track_res: dict[str, Any],
    rr_res: dict[str, Any],
    gate_res: dict[str, Any],
    supply: dict[str, Any],
    df_ind,
) -> dict[str, Any]:
    track_code = track_res["track"]
    suffix = _TRACK_SHORT_SUFFIX.get(track_code, "X")
    sig_id = f"{signal_date.replace('-', '')}_{code}_{suffix}"

    last_obv = float(df_ind["obv"].iloc[-1])
    last_obv_ma5 = float(df_ind["obv_ma5"].iloc[-1])

    return {
        "signal_id": sig_id,
        "signal_date": signal_date,
        "code": code,
        "name": name,
        "track": track_code,
        "track_label": track_res.get("track_label", track_code),
        "entry_price": rr_res["entry_price"],
        "rr_geometry": {
            "invalidation_ref_price": rr_res["invalidation_ref_price"],
            "risk_pct": rr_res["risk_pct"],
            "nearest_resistance_price": rr_res["nearest_resistance_price"],
            "clear_upside_pct": rr_res["clear_upside_pct"],
            "projected_rr": rr_res["projected_rr"],
            "is_blue_sky": bool(rr_res.get("is_blue_sky", False)),
        },
        "snapshot": {
            "chg_pct": gate_res["metrics"]["chg_pct"],
            "marcap_uk": gate_res["metrics"]["marcap_uk"],
            "trading_val_uk": gate_res["metrics"]["trading_val_uk"],
            "atr14_pct": gate_res["metrics"]["atr14_pct"],
            "rs_20d_diff": gate_res["metrics"]["rs_20d_diff"],
            "rs_60d_diff": gate_res["metrics"]["rs_60d_diff"],
            "vol_ratio_20d": gate_res["metrics"]["vol_ratio_20d"],
            "upper_wick_ratio": gate_res["metrics"]["upper_wick_ratio"],
            "frgn_today": int(supply.get("frgn_today", 0)),
            "orgn_today": int(supply.get("orgn_today", 0)),
            "frgn_3d": int(supply.get("frgn_3d", 0)),
            "orgn_3d": int(supply.get("orgn_3d", 0)),
            "obv_trend": "우상향" if last_obv > last_obv_ma5 else "중립",
            "market_regime": gate_res["market_regime"],
            "regime_reason": gate_res["regime_info"]["reason"],
        },
        "tracking": {
            "status": "ACTIVE",
            "days_elapsed": 0,
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "max_r_multiple": 0.0,
            "hit_10pct": False,
            "hit_20pct": False,
            "hit_30pct": False,
            "hit_3r": False,
            "closes": {"d1": None, "d3": None, "d5": None, "d7": None, "d10": None},
        },
        "error_tags": [],
    }


def run_scan(
    provider: MarketDataProvider,
    storage: StorageRepo | None = None,
    notifier: TelegramNotifier | None = None,
    now_kst: datetime | None = None,
    dry_run: bool = False,
    min_trading_val_override: int | None = None,
    send_notifications: bool = True,
) -> dict[str, Any]:
    """장중 코스피 1:3 스윙 스캔 실행: 유니버스 -> 안전필터 -> 공통관문 -> Track A/B/C -> R:R 검증."""
    t_start = time.time()
    store = storage or default_repo
    bot = notifier or TelegramNotifier(dry_run=dry_run)
    eff_now = now_kst or datetime.now(tz=KST)
    eff_min_tval = (
        int(min_trading_val_override)
        if min_trading_val_override is not None
        else get_dynamic_min_trading_val_krw(eff_now)
    )

    index_df = provider.get_index_ohlcv(bars=settings.index_bars)
    trading_dates = index_df["date"].astype(str).tolist() if not index_df.empty else []
    signal_date = trading_dates[-1] if trading_dates else eff_now.strftime("%Y-%m-%d")

    universe = provider.get_universe()
    stats: dict[str, int] = {
        "universe_count": len(universe),
        "pre_filtered_out": 0,
        "cooldown_skipped": 0,
        "safety_rejected": 0,
        "gate_rejected": 0,
        "strategy_rejected": 0,
        "rr_rejected": 0,
        "matched_count": 0,
        "new_signals_saved": 0,
    }
    captured_signals: list[dict[str, Any]] = []
    rr_rejections: list[dict[str, Any]] = []

    for item in universe:
        code = str(item["code"]).zfill(6)
        name = str(item["name"]).strip()
        marcap = float(item.get("marcap", 0.0))
        u_tval = float(item.get("trading_val", 0.0))
        u_chg = float(item.get("chg_pct", 0.0))

        # 유니버스 스냅샷 단계 사전 가지치기 (불필요한 API 호출 절감)
        if u_tval > 0 and u_tval < (eff_min_tval * 0.5):
            stats["pre_filtered_out"] += 1
            continue
        if u_chg != 0.0 and (u_chg < -1.0 or u_chg > (settings.max_chg_pct + 1.0)):
            stats["pre_filtered_out"] += 1
            continue

        in_cd, cd_reason = store.is_in_cooldown(
            code=code,
            target_date=signal_date,
            trading_dates=trading_dates,
            cooldown_days=settings.cooldown_trading_days,
        )
        if in_cd:
            stats["cooldown_skipped"] += 1
            logger.debug("Cooldown skip %s (%s): %s", name, code, cd_reason)
            continue

        safety = provider.get_safety_status(code)
        if not safety.get("is_safe", False):
            stats["safety_rejected"] += 1
            logger.debug("Safety skip %s (%s): %s", name, code, safety.get("reject_reason"))
            continue

        live_tval = float(safety.get("trading_val") or u_tval)
        live_price = float(safety.get("price") or item.get("price") or 0.0)

        df_raw = provider.get_daily_ohlcv(code, bars=settings.ohlcv_bars)
        if not df_raw.empty and str(df_raw["date"].iloc[-1]) == signal_date and live_price > 0:
            last_i = df_raw.index[-1]
            df_raw.loc[last_i, "close"] = live_price
            df_raw.loc[last_i, "high"] = max(float(df_raw.loc[last_i, "high"]), live_price)
            low_curr = float(df_raw.loc[last_i, "low"])
            df_raw.loc[last_i, "low"] = min(low_curr, live_price) if low_curr > 0 else live_price
            if live_tval > 0:
                df_raw.loc[last_i, "amount"] = live_tval

        df_ind = compute_indicators(df_raw)
        if df_ind.empty or len(df_ind) < 25:
            stats["gate_rejected"] += 1
            continue

        # 일봉 마지막 날짜가 signal_date와 맞고 실시간 현재가가 있으면 종가/등락 보정
        entry_price = live_price if live_price > 0 else float(df_ind["close"].iloc[-1])
        gate_res = evaluate_common_gate(
            df=df_ind,
            index_df=index_df,
            trading_val=live_tval,
            marcap_krw=marcap,
            now_kst=eff_now,
            min_trading_val_override=min_trading_val_override,
        )
        if not gate_res["passed"]:
            stats["gate_rejected"] += 1
            continue

        supply = provider.get_investor_supply(code, days=settings.supply_days)
        passed_track, _ = _evaluate_priority_tracks(df_ind, supply)
        if passed_track is None:
            stats["strategy_rejected"] += 1
            continue

        rr_res = evaluate_rr_geometry(
            df=df_ind,
            entry_price=entry_price,
            invalidation_raw=float(passed_track["invalidation_raw"]),
            track=passed_track["track"],
            impulse_start_idx=(passed_track.get("metrics") or {}).get("anchor_idx"),
        )
        if not rr_res["passed"]:
            stats["rr_rejected"] += 1
            rr_rejections.append(
                {
                    "code": code,
                    "name": name,
                    "track": passed_track["track"],
                    "reason": rr_res["reject_reason"],
                    "risk_pct": rr_res["risk_pct"],
                    "clear_upside_pct": rr_res["clear_upside_pct"],
                    "projected_rr": rr_res["projected_rr"],
                }
            )
            continue

        stats["matched_count"] += 1
        sig = _build_signal_record(
            signal_date=signal_date,
            code=code,
            name=name,
            track_res=passed_track,
            rr_res=rr_res,
            gate_res=gate_res,
            supply=supply,
            df_ind=df_ind,
        )
        captured_signals.append(sig)

        is_new = True if dry_run else store.upsert_signal(sig)
        if is_new:
            stats["new_signals_saved"] += 1
            if send_notifications:
                bot.send_signal_alert(sig)

    captured_signals.sort(key=lambda s: s["rr_geometry"]["projected_rr"], reverse=True)
    return {
        "signal_date": signal_date,
        "provider": provider.provider_name,
        "stats": stats,
        "signals": captured_signals,
        "rr_rejections": rr_rejections,
        "elapsed_sec": round(time.time() - t_start, 2),
    }
