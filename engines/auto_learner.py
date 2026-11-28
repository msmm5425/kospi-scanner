from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from config.settings import KST, settings
from storage.repo import StorageRepo, repo as default_repo

EXECUTABLE_TRACK_D_PATTERNS: tuple[str, ...] = (
    "BB_EXTREME_SQUEEZE",
    "ORG_STEALTH_5D",
    "MA_CONVERGENCE",
    "GAP_BREAKAWAY",
    "UPPER_SHADOW_ABSORB",
)

MIN_EFFECTIVE_SAMPLES = 8
PHASE4_MIN_COMPLETED = 30
PHASE4_MIN_WINNERS = 15


def _clamp(val: float, lo: float, hi: float, digits: int = 2) -> float:
    return round(max(lo, min(hi, float(val))), digits)


def _is_winner(sig: dict[str, Any]) -> bool:
    tr = sig.get("tracking") or {}
    return bool(tr.get("hit_10pct") or tr.get("hit_3r"))


def _track_group(track_name: str) -> str:
    t = str(track_name or "").upper()
    if t.startswith("TRACK_D"):
        return "TRACK_D"
    return t


def _adjust_wick_and_rr(
    effective_completed: list[dict[str, Any]],
    recent_top20: list[dict[str, Any]],
    params: dict[str, Any],
    adjustments: list[str],
) -> None:
    losses = [s for s in effective_completed if not _is_winner(s)]
    winners = [s for s in effective_completed if _is_winner(s)]
    if len(effective_completed) < MIN_EFFECTIVE_SAMPLES:
        return

    # 1. FAKEOUT_WICK 승패 대칭 검사 -> max_upper_wick_ratio 보정 [0.30, 0.50]
    curr_wick = float(params.get("max_upper_wick_ratio", settings.max_upper_wick_ratio))
    if losses:
        loss_fake_cnt = sum(1 for s in losses if "FAKEOUT_WICK" in (s.get("error_tags") or []))
        loss_fake_rate = loss_fake_cnt / len(losses)
    else:
        loss_fake_cnt = 0
        loss_fake_rate = 0.0

    win_high_wick_cnt = sum(
        1 for s in winners if float((s.get("snapshot") or {}).get("upper_wick_ratio", 0.0)) >= 0.30
    )
    win_high_wick_rate = (win_high_wick_cnt / len(winners)) if winners else 0.0

    wick_near_misses = sum(
        1
        for d in recent_top20
        for m in (d.get("misses") or [])
        if "윗꼬리 과다" in str(m.get("diagnosis", ""))
    )

    if loss_fake_cnt >= 3 and loss_fake_rate >= 0.40 and win_high_wick_rate < (loss_fake_rate * 0.5):
        new_wick = _clamp(curr_wick - 0.02, 0.30, 0.50)
        if new_wick != curr_wick:
            params["max_upper_wick_ratio"] = new_wick
            adjustments.append(
                f"윗꼬리 상한 하향 ({curr_wick:.2f}->{new_wick:.2f}, FAKEOUT_WICK 손실비중 {loss_fake_rate * 100:.0f}%)"
            )
    elif wick_near_misses >= 4 and loss_fake_rate < 0.15 and curr_wick < settings.max_upper_wick_ratio:
        new_wick = _clamp(curr_wick + 0.02, 0.30, 0.50)
        if new_wick != curr_wick:
            params["max_upper_wick_ratio"] = new_wick
            adjustments.append(
                f"윗꼬리 상한 복원 ({curr_wick:.2f}->{new_wick:.2f}, TOP20 근접탈락 {wick_near_misses}건)"
            )

    # 2. RESISTANCE_REJECT 비율 검사 -> min_projected_rr 보정 [min_projected_rr, 3.2]
    curr_rr = float(params.get("min_projected_rr", settings.min_projected_rr))
    res_reject_cnt = sum(1 for s in losses if "RESISTANCE_REJECT" in (s.get("error_tags") or []))
    res_reject_rate = (res_reject_cnt / len(losses)) if losses else 0.0
    if res_reject_cnt >= 3 and res_reject_rate >= 0.35:
        new_rr = _clamp(curr_rr + 0.1, settings.min_projected_rr, 3.2)
        if new_rr != curr_rr:
            params["min_projected_rr"] = new_rr
            adjustments.append(
                f"최소 손익비 상향 (1:{curr_rr:.1f}->1:{new_rr:.1f}, 매물대저항 손실 {res_reject_rate * 100:.0f}%)"
            )
    elif len(effective_completed) >= 15 and res_reject_rate < 0.10 and curr_rr > settings.min_projected_rr:
        new_rr = _clamp(curr_rr - 0.1, settings.min_projected_rr, 3.2)
        if new_rr != curr_rr:
            params["min_projected_rr"] = new_rr
            adjustments.append(f"최소 손익비 정상화 (1:{curr_rr:.1f}->1:{new_rr:.1f})")


def _adjust_weak_regime(
    effective_completed: list[dict[str, Any]],
    params: dict[str, Any],
    adjustments: list[str],
) -> None:
    weak_comp = [
        s
        for s in effective_completed
        if str((s.get("snapshot") or {}).get("market_regime", "NORMAL")).upper() == "WEAK"
    ]
    if len(weak_comp) < 6:
        return

    weak_3r_rate = sum(1 for s in weak_comp if (s.get("tracking") or {}).get("hit_3r")) / len(weak_comp) * 100.0
    weak_losses = [s for s in weak_comp if not _is_winner(s)]
    index_drag_rate = (
        sum(1 for s in weak_losses if "INDEX_DRAG" in (s.get("error_tags") or [])) / len(weak_losses)
        if weak_losses
        else 0.0
    )

    curr_weak_rr = float(params.get("weak_regime_min_rr", settings.weak_regime_min_rr))
    curr_weak_mult = float(params.get("weak_regime_risk_mult", 1.0))

    if weak_3r_rate < 15.0 or index_drag_rate >= 0.40:
        new_rr = _clamp(curr_weak_rr + 0.2, settings.min_projected_rr, 3.6)
        new_mult = _clamp(curr_weak_mult - 0.05, 0.75, 1.0)
        if new_rr != curr_weak_rr or new_mult != curr_weak_mult:
            params["weak_regime_min_rr"] = new_rr
            params["weak_regime_risk_mult"] = new_mult
            adjustments.append(
                f"약세장 방어 강화 (R:R 1:{new_rr:.1f}, 리스크상한 {new_mult:.2f}x | WEAK 3R {weak_3r_rate:.0f}%)"
            )
    elif weak_3r_rate >= 30.0 and (curr_weak_rr > settings.weak_regime_min_rr or curr_weak_mult < 1.0):
        new_rr = _clamp(curr_weak_rr - 0.1, settings.weak_regime_min_rr, 3.6)
        new_mult = _clamp(curr_weak_mult + 0.05, 0.75, 1.0)
        params["weak_regime_min_rr"] = new_rr
        params["weak_regime_risk_mult"] = new_mult
        adjustments.append(
            f"약세장 정상 완화 (R:R 1:{new_rr:.1f}, 리스크상한 {new_mult:.2f}x | WEAK 3R {weak_3r_rate:.0f}%)"
        )


def _adjust_tracks_and_a_dry(
    effective_completed: list[dict[str, Any]],
    recent_top20: list[dict[str, Any]],
    params: dict[str, Any],
    adjustments: list[str],
) -> None:
    if len(effective_completed) < MIN_EFFECTIVE_SAMPLES:
        return

    risk_mults = dict(params.get("track_risk_mult") or {})
    weights = dict(params.get("track_weights") or {})
    disabled: list[str] = []

    main_tracks = ("TRACK_A_VCP", "TRACK_B_PULLBACK", "TRACK_C_TURNAROUND")
    active_main_cnt = 0

    for tr in ("TRACK_A_VCP", "TRACK_B_PULLBACK", "TRACK_C_TURNAROUND", "TRACK_D"):
        tr_comp = [s for s in effective_completed if _track_group(s.get("track", "")) == tr]
        if len(tr_comp) < 6:
            if tr in main_tracks:
                active_main_cnt += 1
            continue

        tr_3r_rate = sum(1 for s in tr_comp if (s.get("tracking") or {}).get("hit_3r")) / len(tr_comp) * 100.0
        tr_avg_mfe = float(np.mean([float((s.get("tracking") or {}).get("mfe_pct", 0.0)) for s in tr_comp]))
        tr_losses = [s for s in tr_comp if not _is_winner(s)]
        early_breach_cnt = sum(
            1 for s in tr_losses if "RISK_BREACHED_EARLY" in (s.get("error_tags") or [])
        )
        early_rate = (early_breach_cnt / len(tr_losses)) if tr_losses else 0.0

        curr_rm = float(risk_mults.get(tr, 1.0))
        if early_breach_cnt >= 2 and early_rate >= 0.40:
            new_rm = _clamp(curr_rm - 0.05, 0.80, 1.0)
            if new_rm != curr_rm:
                risk_mults[tr] = new_rm
                adjustments.append(
                    f"{tr} 1R 허용폭 축소 ({curr_rm:.2f}x->{new_rm:.2f}x, 조기이탈 {early_rate * 100:.0f}%)"
                )
        elif tr_3r_rate >= 35.0 and curr_rm < 1.0:
            new_rm = _clamp(curr_rm + 0.02, 0.80, 1.0)
            risk_mults[tr] = new_rm

        # 실측 3R 달성률 기반 트랙 가중치 [0.70, 1.25]
        weights[tr] = _clamp(0.75 + (tr_3r_rate / 100.0), 0.70, 1.25)

        if tr in main_tracks:
            # 12건 이상에서 극심한 저성과일 때만 일시 정지 (최소 1개 주력 트랙은 항상 유지)
            if len(tr_comp) >= 12 and tr_3r_rate < 10.0 and tr_avg_mfe < 4.0 and active_main_cnt >= 1:
                disabled.append(tr)
                adjustments.append(f"{tr} 일시 유보 (3R {tr_3r_rate:.1f}%, 평균MFE {tr_avg_mfe:.1f}%)")
            else:
                active_main_cnt += 1

    params["track_risk_mult"] = risk_mults
    params["track_weights"] = weights
    params["disabled_tracks"] = disabled

    # TOP20 근접 탈락 역산 -> Track A 거래량 고갈 한계 보정 [0.65, 0.85]
    dry_misses = sum(
        1
        for d in recent_top20
        for m in (d.get("misses") or [])
        if "Track A" in str(m.get("diagnosis", "")) and "거래량 고갈" in str(m.get("diagnosis", ""))
    )
    tr_a_comp = [s for s in effective_completed if s.get("track") == "TRACK_A_VCP"]
    tr_a_3r = (
        sum(1 for s in tr_a_comp if (s.get("tracking") or {}).get("hit_3r")) / len(tr_a_comp) * 100.0
        if tr_a_comp
        else 0.0
    )
    curr_dry = float(params.get("track_a_dry_vol_ratio_max", settings.track_a_dry_vol_ratio_max))
    if dry_misses >= 5 and (not tr_a_comp or tr_a_3r >= 30.0):
        new_dry = _clamp(curr_dry + 0.02, 0.65, 0.85)
        if new_dry != curr_dry:
            params["track_a_dry_vol_ratio_max"] = new_dry
            adjustments.append(
                f"Track A 거래량 고갈 완화 ({curr_dry * 100:.0f}%->{new_dry * 100:.0f}%, TOP20 병목 {dry_misses}건)"
            )
    elif tr_a_comp and len(tr_a_comp) >= 6 and tr_a_3r < 20.0 and curr_dry > settings.track_a_dry_vol_ratio_max:
        params["track_a_dry_vol_ratio_max"] = settings.track_a_dry_vol_ratio_max
        adjustments.append(
            f"Track A 거래량 고갈 기본 복원 ({settings.track_a_dry_vol_ratio_max * 100:.0f}%, Track A 3R {tr_a_3r:.0f}%)"
        )


def _update_track_d_promotions(
    signals: list[dict[str, Any]],
    patterns: list[dict[str, Any]],
    profile: dict[str, Any],
    as_of_date: str,
    adjustments: list[str],
) -> None:
    ref_dt = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    cutoff_28d = (ref_dt - timedelta(days=28)).strftime("%Y-%m-%d")
    recent_patterns = [p for p in patterns if str(p.get("date", "")) >= cutoff_28d]

    counts: dict[str, dict[str, int]] = {}
    for p in recent_patterns:
        viable = 1 if p.get("rr_viable_1to3") else 0
        for t in p.get("tags") or []:
            tag = str(t).upper()
            if tag not in EXECUTABLE_TRACK_D_PATTERNS:
                continue
            slot = counts.setdefault(tag, {"count": 0, "rr_viable": 0})
            slot["count"] += 1
            slot["rr_viable"] += viable

    demoted_until = dict(profile.get("demoted_track_d_until") or {})
    prev_promoted = {str(x).upper() for x in (profile.get("promoted_track_d_patterns") or [])}
    next_promoted: list[str] = []

    for pat in EXECUTABLE_TRACK_D_PATTERNS:
        cooldown_until = str(demoted_until.get(pat) or "")
        if cooldown_until and as_of_date < cooldown_until:
            continue

        # 이미 실전 승격된 Track D 성적 평가 (>= 8건 완료 후 3R < 20%면 자동 강등)
        td_comp = [
            s
            for s in signals
            if str(s.get("track", "")).upper() == f"TRACK_D_{pat}"
            and (s.get("tracking") or {}).get("status") == "COMPLETED"
        ]
        if len(td_comp) >= 8:
            td_3r = sum(1 for s in td_comp if (s.get("tracking") or {}).get("hit_3r")) / len(td_comp) * 100.0
            if td_3r < 20.0:
                demoted_until[pat] = (ref_dt + timedelta(days=14)).strftime("%Y-%m-%d")
                if pat in prev_promoted:
                    adjustments.append(f"Track D [{pat}] 자동 강등 (실전 3R {td_3r:.1f}% < 20%)")
                continue
            next_promoted.append(pat)
            continue

        meta = counts.get(pat) or {"count": 0, "rr_viable": 0}
        viable_rate = (meta["rr_viable"] / meta["count"]) if meta["count"] > 0 else 0.0
        if meta["count"] >= 4 and meta["rr_viable"] >= 3 and viable_rate >= 0.60:
            next_promoted.append(pat)
            if pat not in prev_promoted:
                adjustments.append(
                    f"Track D [{pat}] 자동 실전 승격 ({meta['count']}회 포착, 1:3 R:R 비율 {viable_rate * 100:.0f}%)"
                )
        elif pat in prev_promoted and meta["count"] >= 2:
            # 최근 표본이 급감하지 않은 한 유지
            next_promoted.append(pat)

    profile["promoted_track_d_patterns"] = next_promoted
    profile["demoted_track_d_until"] = demoted_until


def _update_phase4_policy(
    effective_completed: list[dict[str, Any]],
    profile: dict[str, Any],
    as_of_date: str,
    adjustments: list[str],
) -> bool:
    winners = [s for s in effective_completed if _is_winner(s)]
    phase4 = dict(profile.get("phase4_policy") or {})
    was_active = bool(phase4.get("phase4_active", False))

    if len(effective_completed) < PHASE4_MIN_COMPLETED or len(winners) < PHASE4_MIN_WINNERS:
        profile["phase4_policy"] = phase4
        return False

    winner_maes = [float((s.get("tracking") or {}).get("mae_pct", 0.0)) for s in winners]
    winner_mfes = [float((s.get("tracking") or {}).get("mfe_pct", 0.0)) for s in winners]

    raw_sl = float(np.percentile(winner_maes, 15))
    default_sl = _clamp(raw_sl, -6.5, -3.0)

    sl_by_track: dict[str, float] = {}
    for tr in ("TRACK_A_VCP", "TRACK_B_PULLBACK", "TRACK_C_TURNAROUND", "TRACK_D"):
        tr_win_maes = [
            float((s.get("tracking") or {}).get("mae_pct", 0.0))
            for s in winners
            if _track_group(s.get("track", "")) == tr
        ]
        if len(tr_win_maes) >= 6:
            sl_by_track[tr] = _clamp(float(np.percentile(tr_win_maes, 15)), -6.5, -3.0)

    tp1 = _clamp(max(10.0, float(np.percentile(winner_mfes, 25))), 10.0, 15.0, digits=1)
    tp2 = _clamp(max(tp1 + 5.0, float(np.percentile(winner_mfes, 50))), 16.0, 25.0, digits=1)
    tp3 = _clamp(max(tp2 + 8.0, float(np.percentile(winner_mfes, 75))), 25.0, 40.0, digits=1)

    phase4.update(
        {
            "phase4_active": True,
            "activated_date": phase4.get("activated_date") or as_of_date,
            "default_sl_pct": default_sl,
            "sl_pct_by_track": sl_by_track,
            "tp_levels_pct": [tp1, tp2, tp3],
            "breakeven_trigger_pct": tp1,
        }
    )
    profile["phase4_policy"] = phase4

    just_activated = not was_active
    if just_activated:
        adjustments.append(
            f"Phase 4 자동 전환 (SL {default_sl:+.2f}%, 분할TP +{tp1:.1f}%/+{tp2:.1f}%/+{tp3:.1f}%)"
        )
    return just_activated


def run_auto_learning(
    storage: StorageRepo | None = None,
    as_of_date: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """매일 장 마감 후 오답노트·TOP 20 데이터를 분석해 안전 범위 내 파라미터/Track D/Phase 4 자동 갱신."""
    store = storage or default_repo
    profile = store.load_learning_profile()
    signals = store.get_signals()
    top20_doc = store.load_top20_doc()
    pattern_doc = store.load_pattern_doc()

    eff_date = as_of_date or datetime.now(tz=KST).strftime("%Y-%m-%d")
    start_date = str(profile.get("learning_started_date") or eff_date)
    try:
        d_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        d_now = datetime.strptime(eff_date, "%Y-%m-%d").date()
        learning_days = max(0, (d_now - d_start).days)
    except ValueError:
        learning_days = int(profile.get("learning_days", 0))

    live_signals = [s for s in signals if str(s.get("origin") or "LIVE").upper() != "BOOTSTRAP"]
    bootstrap_signals = [s for s in signals if str(s.get("origin") or "").upper() == "BOOTSTRAP"]

    live_completed = [s for s in live_signals if (s.get("tracking") or {}).get("status") == "COMPLETED"]
    bootstrap_completed = [
        s for s in bootstrap_signals if (s.get("tracking") or {}).get("status") == "COMPLETED"
    ]
    live_winners = [s for s in live_completed if _is_winner(s)]

    # 실전 표본이 20건 이상 쌓이면 부트스트랩 표본 의존을 종료하고 100% 실전 표본으로 전환
    if len(live_completed) >= 20:
        effective_completed = live_completed
        cohort_mode = "LIVE_ONLY"
    else:
        effective_completed = live_completed + bootstrap_completed
        cohort_mode = "LIVE_PLUS_BOOTSTRAP"

    effective_winners = [s for s in effective_completed if _is_winner(s)]

    ref_dt = datetime.strptime(eff_date, "%Y-%m-%d").date()
    cutoff_14d = (ref_dt - timedelta(days=14)).strftime("%Y-%m-%d")
    recent_top20 = [
        d for d in (top20_doc.get("history") or []) if str(d.get("date", "")) >= cutoff_14d
    ]

    adjustments: list[str] = []
    params = dict(profile.get("adaptive_params") or {})

    _adjust_wick_and_rr(effective_completed, recent_top20, params, adjustments)
    _adjust_weak_regime(effective_completed, params, adjustments)
    _adjust_tracks_and_a_dry(effective_completed, recent_top20, params, adjustments)
    profile["adaptive_params"] = params

    _update_track_d_promotions(
        signals=signals,
        patterns=pattern_doc.get("patterns") or [],
        profile=profile,
        as_of_date=eff_date,
        adjustments=adjustments,
    )
    just_activated_phase4 = _update_phase4_policy(
        effective_completed=effective_completed,
        profile=profile,
        as_of_date=eff_date,
        adjustments=adjustments,
    )

    win_3r_rate = (
        round(
            sum(1 for s in effective_completed if (s.get("tracking") or {}).get("hit_3r"))
            / len(effective_completed)
            * 100.0,
            1,
        )
        if effective_completed
        else 0.0
    )
    avg_mfe = (
        round(
            float(
                np.mean([float((s.get("tracking") or {}).get("mfe_pct", 0.0)) for s in effective_completed])
            ),
            2,
        )
        if effective_completed
        else 0.0
    )

    profile["learning_days"] = learning_days
    profile["sample_counts"] = {
        "live_total": len(live_signals),
        "live_completed": len(live_completed),
        "live_winners": len(live_winners),
        "bootstrap_completed": len(bootstrap_completed),
        "effective_completed": len(effective_completed),
        "effective_winners": len(effective_winners),
        "cohort_mode": cohort_mode,
    }
    profile["last_adjustments"] = adjustments

    history = list(profile.get("daily_history") or [])
    daily_entry = {
        "date": eff_date,
        "cohort_mode": cohort_mode,
        "effective_completed": len(effective_completed),
        "effective_winners": len(effective_winners),
        "win_3r_rate": win_3r_rate,
        "avg_mfe": avg_mfe,
        "promoted_track_d": list(profile.get("promoted_track_d_patterns") or []),
        "phase4_active": bool((profile.get("phase4_policy") or {}).get("phase4_active")),
        "adjustments": adjustments,
    }
    replaced = False
    for idx, item in enumerate(history):
        if item.get("date") == eff_date:
            history[idx] = daily_entry
            replaced = True
            break
    if not replaced:
        history.append(daily_entry)
    history.sort(key=lambda x: str(x.get("date", "")))
    profile["daily_history"] = history[-90:]

    if not dry_run:
        store.save_learning_profile(profile)

    return {
        "date": eff_date,
        "learning_days": learning_days,
        "cohort_mode": cohort_mode,
        "sample_counts": profile["sample_counts"],
        "win_3r_rate": win_3r_rate,
        "avg_mfe": avg_mfe,
        "adjustments": adjustments,
        "promoted_track_d_patterns": profile["promoted_track_d_patterns"],
        "phase4_policy": profile["phase4_policy"],
        "just_activated_phase4": just_activated_phase4,
        "adaptive_params": profile["adaptive_params"],
    }
