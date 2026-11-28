from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import subprocess
import threading
import time
from datetime import datetime
from typing import Any

from config.settings import KST, classify_market_session, get_dynamic_min_trading_val_krw, settings
from indicators.technicals import compute_indicators
from notify.telegram_bot import TelegramNotifier
from providers.base import MarketDataProvider
from risk.krx_tick import round_to_tick
from risk.rr_geometry import evaluate_rr_geometry
from storage.repo import StorageRepo, repo as default_repo
from strategies.common_gate import evaluate_common_gate
from strategies.track_a_vcp import evaluate_track_a
from strategies.track_b_pullback import evaluate_track_b
from strategies.track_c_turnaround import evaluate_track_c
from strategies.track_d_learned import evaluate_track_d

logger = logging.getLogger(__name__)

_TRACK_SHORT_SUFFIX = {
    "TRACK_A_VCP": "A",
    "TRACK_B_PULLBACK": "B",
    "TRACK_C_TURNAROUND": "C",
}


def _evaluate_priority_tracks(
    df_ind,
    supply: dict[str, Any],
    adaptive_params: dict[str, Any] | None = None,
    promoted_track_d: list[str] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """우선순위 Track A -> B -> C -> D(자동승격) 순서대로 평가해 첫 통과 트랙 반환."""
    adaptive = adaptive_params or {}
    disabled = {str(t).upper() for t in (adaptive.get("disabled_tracks") or [])}
    dry_max = adaptive.get("track_a_dry_vol_ratio_max")

    diagnostics: list[dict[str, Any]] = []
    if "TRACK_A_VCP" not in disabled:
        res_a = evaluate_track_a(df_ind, supply, dry_vol_ratio_max=dry_max)
        diagnostics.append(res_a)
        if res_a["passed"]:
            return res_a, diagnostics
    if "TRACK_B_PULLBACK" not in disabled:
        res_b = evaluate_track_b(df_ind, supply)
        diagnostics.append(res_b)
        if res_b["passed"]:
            return res_b, diagnostics
    if "TRACK_C_TURNAROUND" not in disabled:
        res_c = evaluate_track_c(df_ind, supply)
        diagnostics.append(res_c)
        if res_c["passed"]:
            return res_c, diagnostics
    if promoted_track_d:
        res_d = evaluate_track_d(df_ind, supply, promoted_patterns=promoted_track_d)
        diagnostics.append(res_d)
        if res_d["passed"]:
            return res_d, diagnostics
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
    origin: str = "LIVE",
    learning_profile: dict[str, Any] | None = None,
    session_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    track_code = track_res["track"]
    suffix = "D" if track_code.startswith("TRACK_D") else _TRACK_SHORT_SUFFIX.get(track_code, "X")
    sig_id = f"{signal_date.replace('-', '')}_{code}_{suffix}"

    last_obv = float(df_ind["obv"].iloc[-1])
    last_obv_ma5 = float(df_ind["obv_ma5"].iloc[-1])
    sess = dict(session_info or {"session": "REGULAR"})

    record: dict[str, Any] = {
        "signal_id": sig_id,
        "signal_date": signal_date,
        "origin": origin,
        "session": str(sess.get("session") or "REGULAR"),
        "code": code,
        "name": name,
        "track": track_code,
        "track_label": track_res.get("track_label", track_code),
        "entry_price": rr_res["entry_price"],
        "max_entry_price": rr_res.get("max_entry_price", rr_res["entry_price"]),
        "session_info": sess,
        "rr_geometry": {
            "invalidation_ref_price": rr_res["invalidation_ref_price"],
            "risk_pct": rr_res["risk_pct"],
            "nearest_resistance_price": rr_res["nearest_resistance_price"],
            "clear_upside_pct": rr_res["clear_upside_pct"],
            "projected_rr": rr_res["projected_rr"],
            "is_blue_sky": bool(rr_res.get("is_blue_sky", False)),
            "max_entry_price": rr_res.get("max_entry_price", rr_res["entry_price"]),
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
    p4 = (learning_profile or {}).get("phase4_policy") or {}
    if p4.get("phase4_active") and p4.get("default_sl_pct") is not None:
        tr_group = "TRACK_D" if track_code.startswith("TRACK_D") else track_code
        sl_pct = float((p4.get("sl_pct_by_track") or {}).get(tr_group) or p4["default_sl_pct"])
        entry_p = float(rr_res["entry_price"])
        sl_price = max(
            int(rr_res["invalidation_ref_price"]),
            round_to_tick(entry_p * (1.0 + sl_pct / 100.0), mode="down"),
        )
        tps = [float(x) for x in (p4.get("tp_levels_pct") or [10.0, 20.0, 30.0])]
        record["phase4_sl_tp"] = {
            "sl_pct": round(sl_pct, 2),
            "sl_price": sl_price,
            "tp_levels": [
                {"pct": tp_pct, "price": round_to_tick(entry_p * (1.0 + tp_pct / 100.0), mode="nearest")}
                for tp_pct in tps
            ],
            "breakeven_trigger_pct": float(p4.get("breakeven_trigger_pct") or tps[0]),
        }
    return record


def run_scan(
    provider: MarketDataProvider,
    storage: StorageRepo | None = None,
    notifier: TelegramNotifier | None = None,
    now_kst: datetime | None = None,
    dry_run: bool = False,
    min_trading_val_override: int | None = None,
    send_notifications: bool = True,
    origin: str = "LIVE",
    learning_profile_override: dict[str, Any] | None = None,
    session_override: str | None = None,
) -> dict[str, Any]:
    """장전/정규장/장후 코스피 1:3 스윙 스캔 실행 (프리·애프터장도 정규장 기준 타점 산출)."""
    t_start = time.time()
    store = storage or default_repo
    bot = notifier or TelegramNotifier(dry_run=dry_run)
    eff_now = now_kst or datetime.now(tz=KST)
    today_kst = eff_now.strftime("%Y-%m-%d")
    session_type = (session_override or classify_market_session(eff_now)).upper()

    learning_profile = learning_profile_override or store.load_learning_profile()
    adaptive = learning_profile.get("adaptive_params") or {}
    promoted_track_d = learning_profile.get("promoted_track_d_patterns") or []
    eff_min_tval = (
        int(min_trading_val_override)
        if min_trading_val_override is not None
        else get_dynamic_min_trading_val_krw(eff_now)
    )

    index_df = provider.get_index_ohlcv(bars=settings.index_bars)
    trading_dates = index_df["date"].astype(str).tolist() if not index_df.empty else []
    # 프리장은 당일 정규장 개장 전이므로 오늘 날짜로 시그널 ID·쿨다운 관리 (전략 평가는 T-1 정규장 일봉 기준)
    if session_type == "PREMARKET":
        signal_date = today_kst
        if signal_date not in trading_dates:
            trading_dates = [*trading_dates, signal_date]
    else:
        signal_date = trading_dates[-1] if trading_dates else today_kst

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
    near_misses: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for item in universe:
        code = str(item["code"]).zfill(6)
        name = str(item["name"]).strip()
        marcap = float(item.get("marcap", 0.0))
        u_tval = float(item.get("trading_val", 0.0))
        u_chg = float(item.get("chg_pct", 0.0))

        # 유니버스 스냅샷 단계 사전 가지치기 (불필요한 API 호출 절감)
        tval_floor = eff_min_tval * (0.85 if session_type == "AFTERMARKET" else 0.5)
        if u_tval > 0 and u_tval < tval_floor:
            stats["pre_filtered_out"] += 1
            continue
        if session_type == "AFTERMARKET" and u_chg != 0.0 and (
            u_chg < (settings.min_chg_pct - 0.5) or u_chg > (settings.max_chg_pct + 0.5)
        ):
            stats["pre_filtered_out"] += 1
            continue
        if session_type == "REGULAR" and u_chg != 0.0 and (u_chg < -1.0 or u_chg > (settings.max_chg_pct + 1.0)):
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

        candidates.append({"code": code, "name": name, "marcap": marcap, "u_tval": u_tval, "u_price": float(item.get("price") or 0.0)})

    def _inspect_candidate(cand: dict[str, Any]) -> dict[str, Any]:
        code = cand["code"]
        name = cand["name"]
        marcap = cand["marcap"]
        u_tval = cand["u_tval"]

        safety = provider.get_safety_status(code)
        if not safety.get("is_safe", False):
            logger.debug("Safety skip %s (%s): %s", name, code, safety.get("reject_reason"))
            return {"outcome": "safety_rejected"}

        if session_type in {"REGULAR", "AFTERMARKET"}:
            s_tval = float(safety.get("trading_val") or 0.0)
            s_chg = float(safety.get("chg_pct") or 0.0)
            if s_tval > 0 and s_tval < (eff_min_tval * 0.95):
                return {"outcome": "gate_rejected"}
            if s_chg != 0.0 and (s_chg < (settings.min_chg_pct - 0.15) or s_chg > (settings.max_chg_pct + 0.15)):
                return {"outcome": "gate_rejected"}

        df_raw = provider.get_daily_ohlcv(code, bars=settings.ohlcv_bars)
        if session_type == "PREMARKET" and not df_raw.empty:
            # 프리장은 직전 완료 정규장(T-1) 일봉만으로 전략·타점 검증
            df_raw = df_raw[df_raw["date"] < today_kst].reset_index(drop=True)

        if session_type == "REGULAR" and not df_raw.empty:
            live_tval = float(safety.get("trading_val") or u_tval)
            live_price = float(safety.get("price") or cand["u_price"] or 0.0)
            if str(df_raw["date"].iloc[-1]) == signal_date and live_price > 0:
                last_i = df_raw.index[-1]
                df_raw.loc[last_i, "close"] = live_price
                df_raw.loc[last_i, "high"] = max(float(df_raw.loc[last_i, "high"]), live_price)
                low_curr = float(df_raw.loc[last_i, "low"])
                df_raw.loc[last_i, "low"] = min(low_curr, live_price) if low_curr > 0 else live_price
                if live_tval > 0:
                    df_raw.loc[last_i, "amount"] = live_tval
        else:
            live_tval = float(df_raw["amount"].iloc[-1]) if not df_raw.empty else u_tval
            live_price = float(df_raw["close"].iloc[-1]) if not df_raw.empty else float(safety.get("price") or 0.0)

        df_ind = compute_indicators(df_raw)
        if df_ind.empty or len(df_ind) < 25:
            return {"outcome": "gate_rejected"}

        # 프리장/애프터장에서도 entry_price 및 모든 기술 지표는 정규장 일봉 기준 고정
        entry_price = (
            live_price
            if (session_type == "REGULAR" and live_price > 0)
            else float(df_ind["close"].iloc[-1])
        )
        eval_tval = (
            live_tval
            if (session_type == "REGULAR" and live_tval > 0)
            else float(df_ind["amount"].iloc[-1])
        )
        gate_res = evaluate_common_gate(
            df=df_ind,
            index_df=index_df,
            trading_val=eval_tval,
            marcap_krw=marcap,
            now_kst=eff_now,
            min_trading_val_override=min_trading_val_override,
            max_upper_wick_override=adaptive.get("max_upper_wick_ratio"),
            max_disparity_override=adaptive.get("max_disparity_20_pct"),
        )
        if not gate_res["passed"]:
            return {"outcome": "gate_rejected"}

        supply = provider.get_investor_supply(code, days=settings.supply_days)
        passed_track, track_diags = _evaluate_priority_tracks(
            df_ind,
            supply,
            adaptive_params=adaptive,
            promoted_track_d=promoted_track_d,
        )
        if passed_track is None:
            best_diag = max(
                track_diags,
                key=lambda d: (
                    int(d.get("passed_count", 0)) / max(1, int(d.get("total_checks", 1))),
                    int(d.get("passed_count", 0)),
                ),
                default=None,
            )
            near_item = None
            if best_diag and int(best_diag.get("passed_count", 0)) >= max(1, int(best_diag.get("total_checks", 1)) - 2):
                near_item = {
                    "code": code,
                    "name": name,
                    "price": int(round(entry_price)),
                    "chg_pct": gate_res["metrics"]["chg_pct"],
                    "track": best_diag.get("track_label") or best_diag.get("track"),
                    "passed_count": int(best_diag.get("passed_count", 0)),
                    "total_checks": int(best_diag.get("total_checks", 1)),
                    "reasons": best_diag.get("failed_reasons") or [],
                }
            return {"outcome": "strategy_rejected", "near_miss": near_item}

        tr_group = "TRACK_D" if passed_track["track"].startswith("TRACK_D") else passed_track["track"]
        t_risk_mult = float((adaptive.get("track_risk_mult") or {}).get(tr_group, 1.0))
        base_min_rr = float(adaptive.get("min_projected_rr", settings.min_projected_rr))
        if gate_res["market_regime"] == "WEAK":
            eff_min_rr = max(base_min_rr, float(adaptive.get("weak_regime_min_rr", settings.weak_regime_min_rr)))
            eff_risk_mult = t_risk_mult * float(adaptive.get("weak_regime_risk_mult", 1.0))
        else:
            eff_min_rr = base_min_rr
            eff_risk_mult = t_risk_mult

        rr_res = evaluate_rr_geometry(
            df=df_ind,
            entry_price=entry_price,
            invalidation_raw=float(passed_track["invalidation_raw"]),
            track=passed_track["track"],
            impulse_start_idx=(passed_track.get("metrics") or {}).get("anchor_idx"),
            min_rr_override=eff_min_rr,
            risk_mult=eff_risk_mult,
        )
        if not rr_res["passed"]:
            return {
                "outcome": "rr_rejected",
                "rr_rejection": {
                    "code": code,
                    "name": name,
                    "track": passed_track["track"],
                    "reason": rr_res["reject_reason"],
                    "risk_pct": rr_res["risk_pct"],
                    "clear_upside_pct": rr_res["clear_upside_pct"],
                    "projected_rr": rr_res["projected_rr"],
                },
            }

        session_info: dict[str, Any] = {
            "session": session_type,
            "captured_time_kst": eff_now.strftime("%H:%M KST"),
            "regular_ref_date": str(df_ind["date"].iloc[-1]),
            "regular_entry_price": rr_res["entry_price"],
            "max_entry_price": rr_res["max_entry_price"],
            "ext_ref_price": None,
            "ext_gap_vs_regular_pct": None,
            "ext_market_code": None,
        }
        if session_type in {"PREMARKET", "AFTERMARKET"}:
            ext_q = provider.get_session_quote(code, session=session_type)
            if ext_q and float(ext_q.get("price") or 0) > 0:
                ext_p = float(ext_q["price"])
                reg_p = float(rr_res["entry_price"])
                gap_vs_reg = round(((ext_p - reg_p) / reg_p) * 100.0, 2) if reg_p > 0 else 0.0
                session_info.update(
                    {
                        "ext_ref_price": int(round(ext_p)),
                        "ext_gap_vs_regular_pct": gap_vs_reg,
                        "ext_market_code": ext_q.get("market_code"),
                    }
                )
                max_p = float(rr_res["max_entry_price"])
                # 프리장에서 이미 정규장 1:3 R:R 추격 상한선 이상 과도 갭상승한 경우 사전 배제
                if session_type == "PREMARKET" and ext_p > max_p * (
                    1.0 + settings.premarket_max_gap_over_ceiling_pct / 100.0
                ):
                    return {
                        "outcome": "rr_rejected",
                        "rr_rejection": {
                            "code": code,
                            "name": name,
                            "track": passed_track["track"],
                            "reason": f"프리장 갭과열 ({int(ext_p):,}원 > 정규장 상한 {int(max_p):,}원)",
                            "risk_pct": rr_res["risk_pct"],
                            "clear_upside_pct": rr_res["clear_upside_pct"],
                            "projected_rr": rr_res["projected_rr"],
                        },
                    }

        sig = _build_signal_record(
            signal_date=signal_date,
            code=code,
            name=name,
            track_res=passed_track,
            rr_res=rr_res,
            gate_res=gate_res,
            supply=supply,
            df_ind=df_ind,
            origin=origin,
            learning_profile=learning_profile,
            session_info=session_info,
        )
        return {"outcome": "matched", "signal": sig}

    max_workers = 1 if (len(candidates) <= 1 or provider.provider_name in {"mock", "bootstrap_slice"}) else min(6, len(candidates))
    if max_workers <= 1:
        results = [_inspect_candidate(c) for c in candidates]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_inspect_candidate, candidates))

    for out in results:
        kind = out["outcome"]
        if kind == "safety_rejected":
            stats["safety_rejected"] += 1
        elif kind == "gate_rejected":
            stats["gate_rejected"] += 1
        elif kind == "strategy_rejected":
            stats["strategy_rejected"] += 1
            if out.get("near_miss"):
                near_misses.append(out["near_miss"])
        elif kind == "rr_rejected":
            stats["rr_rejected"] += 1
            rr_rejections.append(out["rr_rejection"])
        elif kind == "matched":
            stats["matched_count"] += 1
            sig = out["signal"]
            captured_signals.append(sig)
            is_new = True if dry_run else store.upsert_signal(sig)
            if is_new:
                stats["new_signals_saved"] += 1
                if send_notifications:
                    bot.send_signal_alert(sig)

    captured_signals.sort(key=lambda s: s["rr_geometry"]["projected_rr"], reverse=True)
    near_misses.sort(
        key=lambda m: (m["passed_count"] / max(1, m["total_checks"]), m["passed_count"], m["chg_pct"]),
        reverse=True,
    )
    return {
        "signal_date": signal_date,
        "session": session_type,
        "provider": provider.provider_name,
        "stats": stats,
        "signals": captured_signals,
        "rr_rejections": rr_rejections,
        "near_misses": near_misses[:10],
        "elapsed_sec": round(time.time() - t_start, 2),
    }


class _LoopSessionCacheProvider(MarketDataProvider):
    """--loop 연속 스캔 중 과거 150일 일봉·확정수급·지수를 캐시해 API 한도·속도를 최적화하는 래퍼."""

    def __init__(self, inner: MarketDataProvider):
        self._inner = inner
        self.provider_name = inner.provider_name
        self._lock = threading.RLock()
        self._last_session = ""
        self._last_date = ""
        self._index_cache: tuple[float, Any] | None = None
        self._ohlcv_cache: dict[str, tuple[float, Any]] = {}
        self._supply_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def _maybe_reset_on_transition(self, eff_now: datetime) -> None:
        curr_date = eff_now.strftime("%Y-%m-%d")
        curr_sess = classify_market_session(eff_now)
        with self._lock:
            if curr_date != self._last_date or curr_sess != self._last_session:
                self._index_cache = None
                self._ohlcv_cache.clear()
                self._supply_cache.clear()
                self._last_date = curr_date
                self._last_session = curr_sess

    def _ttl(self) -> float:
        return 900.0 if self._last_session == "REGULAR" else 3600.0

    def get_universe(self, min_cap=None, max_cap=None):
        return self._inner.get_universe(min_cap=min_cap, max_cap=max_cap)

    def get_safety_status(self, code: str):
        return self._inner.get_safety_status(code)

    def get_daily_ohlcv(self, code: str, bars: int = 150, end_date: str | None = None):
        now_ts = time.time()
        with self._lock:
            cached = self._ohlcv_cache.get(code)
            if cached and (now_ts - cached[0]) < self._ttl():
                return cached[1].copy()
        df = self._inner.get_daily_ohlcv(code, bars=bars, end_date=end_date)
        with self._lock:
            self._ohlcv_cache[code] = (now_ts, df)
        return df.copy()

    def get_investor_supply(self, code: str, days: int = 5, end_date: str | None = None):
        now_ts = time.time()
        with self._lock:
            cached = self._supply_cache.get(code)
            if cached and (now_ts - cached[0]) < max(self._ttl(), 1800.0):
                return dict(cached[1])
        sup = self._inner.get_investor_supply(code, days=days, end_date=end_date)
        with self._lock:
            self._supply_cache[code] = (now_ts, sup)
        return dict(sup)

    def get_index_ohlcv(self, bars: int = 100, end_date: str | None = None):
        now_ts = time.time()
        with self._lock:
            if self._index_cache and (now_ts - self._index_cache[0]) < 300.0:
                return self._index_cache[1].copy()
        df = self._inner.get_index_ohlcv(bars=bars, end_date=end_date)
        with self._lock:
            self._index_cache = (now_ts, df)
        return df.copy()

    def get_top_gainers(self, top_n: int = 20, end_date: str | None = None):
        return self._inner.get_top_gainers(top_n=top_n, end_date=end_date)

    def get_sector_map(self):
        return self._inner.get_sector_map()

    def is_trading_day(self, date_str: str | None = None) -> bool:
        return self._inner.is_trading_day(date_str)

    def get_session_quote(self, code: str, session: str = "REGULAR"):
        return self._inner.get_session_quote(code, session=session)


def _git_sync_pull() -> None:
    try:
        subprocess.run(
            ["git", "pull", "--rebase", "--autostash", "origin", "main"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:
        logger.debug("git pull rebase skipped: %s", e)


def _git_sync_commit_signals(store: StorageRepo, new_signals: list[dict[str, Any]]) -> None:
    """신규 포착 시그널 및 브리핑 상태를 Git에 커밋·푸시하고, 15:50 분석과 경합 시 안전 병합한다."""
    try:
        subprocess.run(
            ["git", "add", str(store.signals_path), str(store.learning_path)],
            check=False,
            timeout=15,
        )
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False, timeout=15)
        if diff.returncode == 0:
            return
        subprocess.run(
            ["git", "commit", "-m", "chore(data): update signals_log [skip ci]"],
            check=False,
            capture_output=True,
            timeout=15,
        )
        for _ in range(3):
            pull = subprocess.run(
                ["git", "pull", "--rebase", "origin", "main"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if pull.returncode != 0:
                subprocess.run(["git", "rebase", "--abort"], check=False, timeout=15)
                subprocess.run(["git", "fetch", "origin", "main"], check=False, timeout=20)
                subprocess.run(["git", "reset", "--hard", "origin/main"], check=False, timeout=20)
                for sig in new_signals:
                    store.upsert_signal(sig)
                subprocess.run(
                    ["git", "add", str(store.signals_path), str(store.learning_path)],
                    check=False,
                    timeout=15,
                )
                subprocess.run(
                    ["git", "commit", "-m", "chore(data): merge signals_log [skip ci]"],
                    check=False,
                    capture_output=True,
                    timeout=15,
                )
            push = subprocess.run(
                ["git", "push", "origin", "HEAD:main"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if push.returncode == 0:
                return
            time.sleep(2.0)
    except Exception as e:
        logger.warning("git sync signals failed cleanly: %s", e)


def run_loop_mode(
    provider: MarketDataProvider,
    storage: StorageRepo | None = None,
    notifier: TelegramNotifier | None = None,
    interval_sec: int = 240,
    max_minutes: int = 260,
    end_kst: str = "21:05",
    dry_run: bool = False,
    send_notifications: bool = True,
    git_sync: bool = False,
    force: bool = False,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    """08:00~21:00 연속 루프 실행 (공휴일 조기 종료, 캐시 재사용, 자동 Git 동기화)."""
    store = storage or default_repo
    bot = notifier or TelegramNotifier(dry_run=dry_run)
    cached_provider = _LoopSessionCacheProvider(provider)

    start_wall = time.time()
    max_budget_sec = max(10.0, float(max_minutes) * 60.0)
    end_h, end_m = [int(x) for x in end_kst.split(":", 1)]

    now_init = datetime.now(tz=KST)
    if not force and not cached_provider.is_trading_day(now_init.strftime("%Y-%m-%d")):
        print(f"[LOOP 종료] {now_init.strftime('%Y-%m-%d')}은(는) KRX 휴장일입니다.")
        return {"cycles": 0, "total_new_signals": 0, "skipped_holiday": True}

    cycles = 0
    total_new = 0
    while True:
        now_kst = datetime.now(tz=KST)
        if not force and (now_kst.hour > end_h or (now_kst.hour == end_h and now_kst.minute >= end_m)):
            print(f"[LOOP 종료] 종료 시각({end_kst} KST) 도달.")
            break
        if (time.time() - start_wall) >= max_budget_sec:
            print(f"[LOOP 종료] 세션 실행 예산({max_minutes}분) 도달 — 다음 릴레이에 인계.")
            break
        if max_iterations is not None and cycles >= max_iterations:
            break

        if git_sync:
            _git_sync_pull()

        cached_provider._maybe_reset_on_transition(now_kst)
        res = run_scan(
            provider=cached_provider,
            storage=store,
            notifier=bot,
            now_kst=now_kst,
            dry_run=dry_run,
            send_notifications=send_notifications,
        )
        cycles += 1
        new_cnt = int(res["stats"]["new_signals_saved"])
        total_new += new_cnt
        print(
            f"[LOOP #{cycles}] {now_kst.strftime('%H:%M:%S KST')} ({res['session']}) | "
            f"provider={res['provider']} universe={res['stats']['universe_count']} | "
            f"matched={res['stats']['matched_count']} new={new_cnt} ({res['elapsed_sec']}s)"
        )

        # 매일 15:00~15:18 KST 도달 시 1회 장 마감 전 최종 매수 픽 브리핑 자동 실행
        brief_sent_now = False
        if now_kst.hour == 15 and 0 <= now_kst.minute < 18:
            from engines.closing_picker import build_and_send_closing_briefing

            brief = build_and_send_closing_briefing(
                provider=cached_provider,
                storage=store,
                notifier=bot,
                now_kst=now_kst,
                dry_run=dry_run,
                send_notifications=send_notifications,
            )
            if not brief.get("already_sent"):
                brief_sent_now = True
                print(
                    f"[15:00 종가 브리핑] qualified={len(brief.get('qualified') or [])} | "
                    f"invalidated={len(brief.get('invalidated') or [])}"
                )

        if git_sync and (new_cnt > 0 or brief_sent_now) and not dry_run:
            _git_sync_commit_signals(store, res["signals"])

        if max_iterations is not None and cycles >= max_iterations:
            break
        remaining = max_budget_sec - (time.time() - start_wall)
        if remaining <= 0:
            break
        sleep_for = min(float(interval_sec), max(0.0, remaining - 2.0))
        if sleep_for > 0:
            time.sleep(sleep_for)

    return {
        "cycles": cycles,
        "total_new_signals": total_new,
        "skipped_holiday": False,
    }
