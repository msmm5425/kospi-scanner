from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from config.settings import KST
from notify.telegram_bot import TelegramNotifier
from storage.repo import StorageRepo, repo as default_repo


def _summarize_track(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    completed = [r for r in records if (r.get("tracking") or {}).get("status") == "COMPLETED"]
    n_comp = len(completed)
    # 진행 중이라도 이미 목표 도달한 경우가 있으므로 hit 통계는 전체 중 달성 + 완료 기준 병행 산출
    base = completed if n_comp > 0 else records
    denom = len(base)
    if denom == 0:
        return {
            "total": total,
            "completed": 0,
            "hit_10_rate": 0.0,
            "hit_20_rate": 0.0,
            "hit_30_rate": 0.0,
            "hit_3r_rate": 0.0,
            "avg_mfe": 0.0,
            "avg_mae": 0.0,
            "avg_max_r": 0.0,
        }

    hit10 = sum(1 for r in base if (r.get("tracking") or {}).get("hit_10pct"))
    hit20 = sum(1 for r in base if (r.get("tracking") or {}).get("hit_20pct"))
    hit30 = sum(1 for r in base if (r.get("tracking") or {}).get("hit_30pct"))
    hit3r = sum(1 for r in base if (r.get("tracking") or {}).get("hit_3r"))
    mfes = [float((r.get("tracking") or {}).get("mfe_pct", 0.0)) for r in base]
    maes = [float((r.get("tracking") or {}).get("mae_pct", 0.0)) for r in base]
    max_rs = [float((r.get("tracking") or {}).get("max_r_multiple", 0.0)) for r in base]

    return {
        "total": total,
        "completed": n_comp,
        "hit_10_rate": round(hit10 / denom * 100.0, 1),
        "hit_20_rate": round(hit20 / denom * 100.0, 1),
        "hit_30_rate": round(hit30 / denom * 100.0, 1),
        "hit_3r_rate": round(hit3r / denom * 100.0, 1),
        "avg_mfe": round(float(np.mean(mfes)), 2),
        "avg_mae": round(float(np.mean(maes)), 2),
        "avg_max_r": round(float(np.mean(max_rs)), 2),
    }


def build_weekly_report(
    storage: StorageRepo | None = None,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """트랙별 3R 달성률, 오답 TOP 3, Track D 후보, Phase 4 SL/TP 분포표 산출."""
    store = storage or default_repo
    signals = store.get_signals()
    pattern_doc = store.load_pattern_doc()
    patterns = pattern_doc.get("patterns", [])

    ref_date_str = as_of_date or datetime.now(tz=KST).strftime("%Y-%m-%d")
    ref_dt = datetime.strptime(ref_date_str, "%Y-%m-%d").date()
    week_cutoff = (ref_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    month_cutoff = (ref_dt - timedelta(days=28)).strftime("%Y-%m-%d")

    weekly_signals = [s for s in signals if str(s.get("signal_date", "")) >= week_cutoff]

    # 트랙별 성과
    track_stats: dict[str, Any] = {}
    for tr in ("TRACK_A_VCP", "TRACK_B_PULLBACK", "TRACK_C_TURNAROUND"):
        track_stats[tr] = _summarize_track([s for s in signals if s.get("track") == tr])
    track_stats["ALL"] = _summarize_track(signals)

    # 장세별 성과 비교 (NORMAL vs WEAK)
    regime_stats = {
        "NORMAL": _summarize_track(
            [s for s in signals if (s.get("snapshot") or {}).get("market_regime", "NORMAL") == "NORMAL"]
        ),
        "WEAK": _summarize_track(
            [s for s in signals if (s.get("snapshot") or {}).get("market_regime") == "WEAK"]
        ),
    }

    # 오답 TOP 3
    err_counter: Counter[str] = Counter()
    for s in signals:
        for tag in s.get("error_tags") or []:
            err_counter[str(tag)] += 1
    top3_errors = [{"tag": tag, "count": cnt} for tag, cnt in err_counter.most_common(3)]

    # 최근 4주 pattern_queue -> Track D 승격 후보
    recent_patterns = [p for p in patterns if str(p.get("date", "")) >= month_cutoff]
    tag_stats: dict[str, dict[str, int]] = {}
    for p in recent_patterns:
        viable = 1 if p.get("rr_viable_1to3") else 0
        for t in p.get("tags") or []:
            slot = tag_stats.setdefault(str(t), {"count": 0, "rr_viable": 0})
            slot["count"] += 1
            slot["rr_viable"] += viable
    track_d_candidates = sorted(
        [
            {
                "pattern": tag,
                "count": meta["count"],
                "rr_viable_count": meta["rr_viable"],
                "rr_viable_rate_pct": round(meta["rr_viable"] / meta["count"] * 100.0, 1),
            }
            for tag, meta in tag_stats.items()
        ],
        key=lambda x: (x["rr_viable_count"], x["count"]),
        reverse=True,
    )

    # Phase 4 SL/TP 최적화 분포표 (목표 +10% 또는 3R 달성 표본 기반)
    winners = [
        s
        for s in signals
        if ((s.get("tracking") or {}).get("hit_10pct") or (s.get("tracking") or {}).get("hit_3r"))
    ]
    winner_maes = [float((s.get("tracking") or {}).get("mae_pct", 0.0)) for s in winners]
    winner_mfes = [float((s.get("tracking") or {}).get("mfe_pct", 0.0)) for s in winners]

    # MAE는 음수이므로 85%의 승리 표본이 버틴 하한선 = 15th percentile (예: -4.2%)
    mae_85pct_sl = round(float(np.percentile(winner_maes, 15)), 2) if winner_maes else None
    mfe_dist = (
        {
            "p25": round(float(np.percentile(winner_mfes, 25)), 2),
            "p50": round(float(np.percentile(winner_mfes, 50)), 2),
            "p75": round(float(np.percentile(winner_mfes, 75)), 2),
        }
        if winner_mfes
        else None
    )

    return {
        "as_of_date": ref_date_str,
        "weekly_signal_count": len(weekly_signals),
        "total_signal_count": len(signals),
        "winner_sample_count": len(winners),
        "phase4_ready": len(winners) >= 30,
        "track_stats": track_stats,
        "regime_stats": regime_stats,
        "top3_errors": top3_errors,
        "track_d_candidates": track_d_candidates[:5],
        "phase4_distribution": {
            "recommended_sl_mae_85pct": mae_85pct_sl,
            "mfe_percentiles": mfe_dist,
        },
    }


def run_weekly_report(
    storage: StorageRepo | None = None,
    notifier: TelegramNotifier | None = None,
    as_of_date: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    report = build_weekly_report(storage=storage, as_of_date=as_of_date)
    bot = notifier or TelegramNotifier(dry_run=dry_run)
    bot.send_weekly_report(report)
    return report
