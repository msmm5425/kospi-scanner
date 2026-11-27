from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from config.settings import KST, settings
from indicators.technicals import compute_indicators
from notify.telegram_bot import TelegramNotifier
from providers.base import MarketDataProvider
from risk.krx_tick import round_to_tick
from risk.rr_geometry import _max_risk_limit_for_track
from storage.repo import StorageRepo, repo as default_repo
from strategies.common_gate import evaluate_common_gate

logger = logging.getLogger(__name__)


def _closing_audit_single(
    sig: dict[str, Any],
    provider: MarketDataProvider,
    index_df,
    adaptive: dict[str, Any],
    now_kst: datetime,
) -> dict[str, Any]:
    """15:00 종가 매수 직전 시점에 당일 포착 종목의 현재가·윗꼬리·1:3 손익비 유효성을 재검증한다."""
    code = str(sig.get("code") or "").zfill(6)
    name = str(sig.get("name") or "")
    track = str(sig.get("track") or "")
    rr_orig = sig.get("rr_geometry") or {}
    inval_p = float(rr_orig.get("invalidation_ref_price") or 0.0)
    nearest_res = rr_orig.get("nearest_resistance_price")
    is_blue_sky = bool(rr_orig.get("is_blue_sky", False))
    orig_entry = float(sig.get("entry_price") or 0.0)
    max_entry = float(rr_orig.get("max_entry_price") or sig.get("max_entry_price") or orig_entry * 1.03)

    tr_group = "TRACK_D" if track.startswith("TRACK_D") else track
    t_risk_mult = float((adaptive.get("track_risk_mult") or {}).get(tr_group, 1.0))
    base_min_rr = float(adaptive.get("min_projected_rr", settings.min_projected_rr))

    safety = provider.get_safety_status(code)
    live_price = float(safety.get("price") or orig_entry)
    live_tval = float(safety.get("trading_val") or 0.0)

    df_raw = provider.get_daily_ohlcv(code, bars=settings.ohlcv_bars)
    signal_date = str(sig.get("signal_date") or now_kst.strftime("%Y-%m-%d"))
    if not df_raw.empty and str(df_raw["date"].iloc[-1]) == signal_date and live_price > 0:
        last_i = df_raw.index[-1]
        df_raw.loc[last_i, "close"] = live_price
        df_raw.loc[last_i, "high"] = max(float(df_raw.loc[last_i, "high"]), live_price)
        low_curr = float(df_raw.loc[last_i, "low"])
        df_raw.loc[last_i, "low"] = min(low_curr, live_price) if low_curr > 0 else live_price
        if live_tval > 0:
            df_raw.loc[last_i, "amount"] = live_tval

    df_ind = compute_indicators(df_raw)
    closing_tick = round_to_tick(live_price if live_price > 0 else orig_entry, mode="nearest")

    reasons: list[str] = []
    if not safety.get("is_safe", False):
        reasons.append(f"안전필터 부적격({safety.get('reject_reason', '유의/정지')})")

    gate_res = evaluate_common_gate(
        df=df_ind,
        index_df=index_df,
        trading_val=live_tval if live_tval > 0 else (float(df_ind["amount"].iloc[-1]) if not df_ind.empty else 0.0),
        marcap_krw=float((sig.get("snapshot") or {}).get("marcap_uk", 0.0)) * 1e8,
        now_kst=now_kst,
        max_upper_wick_override=adaptive.get("max_upper_wick_ratio"),
        max_disparity_override=adaptive.get("max_disparity_20_pct"),
    )
    if not gate_res["passed"]:
        reasons.extend(gate_res["failed_reasons"])

    if gate_res["market_regime"] == "WEAK":
        eff_min_rr = max(base_min_rr, float(adaptive.get("weak_regime_min_rr", 3.5)))
        eff_risk_mult = t_risk_mult * float(adaptive.get("weak_regime_risk_mult", 1.0))
    else:
        eff_min_rr = base_min_rr
        eff_risk_mult = t_risk_mult
    max_risk_pct = round(_max_risk_limit_for_track(track) * eff_risk_mult, 2)

    if inval_p > 0 and closing_tick <= inval_p:
        reasons.append(f"무효화선(1R {int(inval_p):,}원) 하향 이탈")

    closing_risk = round(((closing_tick - inval_p) / closing_tick) * 100.0, 2) if (closing_tick > 0 and inval_p > 0) else 99.0
    if closing_risk > max_risk_pct:
        reasons.append(f"15:00 종가 리스크 초과 ({closing_risk:.2f}% > {max_risk_pct:.1f}%)")

    if is_blue_sky or nearest_res is None:
        gap_from_orig = ((closing_tick - orig_entry) / orig_entry * 100.0) if orig_entry > 0 else 0.0
        closing_upside = round(max(0.0, float(rr_orig.get("clear_upside_pct") or 30.0) - max(0.0, gap_from_orig)), 2)
    else:
        closing_upside = round(((float(nearest_res) - closing_tick) / closing_tick) * 100.0, 2) if closing_tick > 0 else 0.0

    if closing_upside < settings.min_clear_upside_pct:
        reasons.append(f"상단 여력 부족 (+{closing_upside:.1f}% < +10%)")

    closing_rr = round(closing_upside / closing_risk, 2) if closing_risk > 0 else 0.0
    if closing_rr < eff_min_rr:
        reasons.append(f"종가 R:R 미달 (1:{closing_rr:.2f} < 1:{eff_min_rr:.1f})")

    if closing_tick > max_entry * 1.01:
        reasons.append(f"추격 상한 초과 ({closing_tick:,}원 > {int(round(max_entry)):,}원)")

    m = gate_res.get("metrics") or {}
    wick_now = float(m.get("upper_wick_ratio", 0.0))
    rs_20_now = float(m.get("rs_20d_diff", 0.0))
    chg_now = float(m.get("chg_pct", safety.get("chg_pct") or 0.0))
    track_w = float((adaptive.get("track_weights") or {}).get(tr_group, 1.0))

    validated = len(reasons) == 0
    if validated:
        score = round(
            (min(closing_rr, 8.0) * 12.0 + max(0.0, 1.0 - wick_now) * 25.0 + min(max(rs_20_now, 0.0), 20.0) * 0.5)
            * track_w,
            1,
        )
    else:
        score = 0.0

    return {
        "code": code,
        "name": name,
        "track": track,
        "track_label": sig.get("track_label", track),
        "signal_id": sig.get("signal_id"),
        "validated": validated,
        "checked_at": now_kst.strftime("%H:%M KST"),
        "closing_price": closing_tick,
        "closing_chg_pct": round(chg_now, 2),
        "closing_wick": round(wick_now, 3),
        "closing_risk_pct": closing_risk,
        "closing_upside_pct": closing_upside,
        "closing_rr": closing_rr,
        "invalidation_ref_price": int(round(inval_p)),
        "max_entry_price": int(round(max_entry)),
        "nearest_resistance_price": int(round(float(nearest_res))) if nearest_res else None,
        "is_blue_sky": is_blue_sky,
        "closing_score": score,
        "invalidate_reasons": reasons,
        "market_regime": gate_res["market_regime"],
    }


def build_and_send_closing_briefing(
    provider: MarketDataProvider,
    storage: StorageRepo | None = None,
    notifier: TelegramNotifier | None = None,
    now_kst: datetime | None = None,
    dry_run: bool = False,
    send_notifications: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """15:00 장 마감 매수 직전 당일 포착 전수 재검증 + TOP 3 종가 베팅 브리핑 생성/발송."""
    store = storage or default_repo
    bot = notifier or TelegramNotifier(dry_run=dry_run)
    eff_now = now_kst or datetime.now(tz=KST)
    today_kst = eff_now.strftime("%Y-%m-%d")

    lp = store.load_learning_profile()
    if not force and str(lp.get("last_closing_briefing_date") or "") == today_kst:
        return {"already_sent": True, "date": today_kst, "qualified": [], "invalidated": []}

    index_df = provider.get_index_ohlcv(bars=settings.index_bars)
    adaptive = lp.get("adaptive_params") or {}
    all_signals = store.get_signals()
    today_signals = [
        s
        for s in all_signals
        if str(s.get("signal_date", "")) == today_kst and str(s.get("origin") or "LIVE").upper() != "BOOTSTRAP"
    ]

    qualified: list[dict[str, Any]] = []
    invalidated: list[dict[str, Any]] = []

    for sig in today_signals:
        audit = _closing_audit_single(sig, provider, index_df, adaptive, eff_now)
        if not dry_run:
            updated = dict(sig)
            updated["closing_audit"] = {
                "validated": audit["validated"],
                "checked_at": audit["checked_at"],
                "closing_price": audit["closing_price"],
                "closing_chg_pct": audit["closing_chg_pct"],
                "closing_wick": audit["closing_wick"],
                "closing_rr": audit["closing_rr"],
                "closing_score": audit["closing_score"],
                "invalidate_reasons": audit["invalidate_reasons"],
            }
            store.upsert_signal(updated)

        if audit["validated"]:
            qualified.append(audit)
        else:
            invalidated.append(audit)

    qualified.sort(key=lambda x: (x["closing_score"], x["closing_rr"]), reverse=True)

    regime = (
        qualified[0]["market_regime"]
        if qualified
        else (invalidated[0]["market_regime"] if invalidated else "NORMAL")
    )
    briefing = {
        "already_sent": False,
        "date": today_kst,
        "checked_at": eff_now.strftime("%H:%M KST"),
        "market_regime": regime,
        "total_today_signals": len(today_signals),
        "qualified": qualified,
        "invalidated": invalidated,
    }

    if not dry_run:
        lp["last_closing_briefing_date"] = today_kst
        store.save_learning_profile(lp)

    if send_notifications:
        bot.send_closing_briefing(briefing)

    return briefing
