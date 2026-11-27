from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from config.settings import KST
from engines.error_log import update_error_logs
from engines.scanner import run_scan
from engines.top20_analyzer import analyze_top20
from notify.telegram_bot import format_signal_message, format_top20_analysis_message
from providers.factory import create_provider
from storage.repo import StorageRepo


def run_synthetic_replay() -> int:
    """네트워크 없이 Phase 1~2 전체 파이프라인 드라이런 검증."""
    from tests.test_core import (
        MockProvider,
        _make_index_df,
        _make_track_a_vcp_df,
        _make_track_b_pullback_df,
        _make_track_c_turnaround_df,
        _trading_dates,
    )

    dates = _trading_dates(160, end="2026-11-20")
    df_a = _make_track_a_vcp_df(dates)
    df_b = _make_track_b_pullback_df(dates)
    df_c = _make_track_c_turnaround_df(dates)

    sup_good = {
        "frgn_today": 12_000,
        "orgn_today": 18_000,
        "frgn_3d": 45_000,
        "orgn_3d": 35_000,
        "total_5d": 95_000,
        "pos_days_5d": 4,
        "intraday_pending": False,
        "orgn_series": [5_000, 6_000, 7_000, 8_000, 10_000],
    }
    provider = MockProvider(
        universe=[
            {"code": "100010", "name": "가상VCP", "marcap": 600_000_000_000, "trading_val": float(df_a.iloc[-1]["amount"]), "price": float(df_a.iloc[-1]["close"]), "chg_pct": 4.5},
            {"code": "100020", "name": "가상눌림목", "marcap": 450_000_000_000, "trading_val": float(df_b.iloc[-1]["amount"]), "price": float(df_b.iloc[-1]["close"]), "chg_pct": 4.5},
            {"code": "100030", "name": "가상매집전환", "marcap": 320_000_000_000, "trading_val": float(df_c.iloc[-1]["amount"]), "price": float(df_c.iloc[-1]["close"]), "chg_pct": 4.5},
        ],
        ohlcv_map={"100010": df_a, "100020": df_b, "100030": df_c},
        supply_map={"100010": sup_good, "100020": sup_good, "100030": sup_good},
        index_df=_make_index_df(dates, weak=False),
        top_gainers=[
            {"code": "100010", "name": "가상VCP", "close": float(df_a.iloc[-1]["close"]), "chg_pct": 12.5, "trading_val": 5e9, "marcap": 6e11}
        ],
    )
    with tempfile.TemporaryDirectory() as tmp:
        repo = StorageRepo(data_dir=Path(tmp))
        scan_res = run_scan(
            provider=provider,
            storage=repo,
            now_kst=datetime(2026, 11, 20, 14, 0, tzinfo=KST),
            send_notifications=False,
        )
        print(f"[SYNTHETIC REPLAY] 포착 {len(scan_res['signals'])}종목:")
        for sig in scan_res["signals"]:
            print("---\n" + format_signal_message(sig))

        err_res = update_error_logs(provider=provider, storage=repo, as_of_date="2026-11-20")
        top20_res = analyze_top20(provider=provider, storage=repo, as_of_date="2026-11-20")
        print("---\n" + format_top20_analysis_message(err_res, top20_res))
    return 0


def run_live_date_replay(target_date: str) -> int:
    """실제 과거 거래일 기준 PyKRX 리플레이 드라이런 (storage 파일 비오염)."""
    provider = create_provider(provider_name="pykrx", as_of_date=target_date)
    with tempfile.TemporaryDirectory() as tmp:
        repo = StorageRepo(data_dir=Path(tmp))
        scan_res = run_scan(
            provider=provider,
            storage=repo,
            now_kst=datetime.strptime(target_date, "%Y-%m-%d").replace(hour=15, minute=10, tzinfo=KST),
            dry_run=True,
            send_notifications=False,
        )
        print(
            f"[LIVE REPLAY] 요청일={target_date} -> 실제거래일={scan_res['signal_date']} | "
            f"유니버스={scan_res['stats']['universe_count']} | 포착={len(scan_res['signals'])}"
        )
        for sig in scan_res["signals"]:
            print("---\n" + format_signal_message(sig))

        top20_res = analyze_top20(
            provider=provider,
            storage=repo,
            as_of_date=target_date,
            dry_run=True,
        )
        print(f"\n[TOP20 역추적] {top20_res['summary']['date']} (MISS {top20_res['summary']['miss_count']}건):")
        for m in top20_res["summary"]["misses"][:5]:
            print(f"  - {m['name']} ({m['chg_pct']:+.1f}%): {m['diagnosis']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="과거 특정 날짜 기준 코스피 1:3 스캐너 리플레이")
    parser.add_argument("--date", default="2026-11-20", help="리플레이 기준 날짜 YYYY-MM-DD")
    parser.add_argument("--synthetic", action="store_true", help="오프라인 합성 데이터 드라이런 실행")
    args = parser.parse_args(argv)
    if args.synthetic:
        return run_synthetic_replay()
    return run_live_date_replay(args.date)


if __name__ == "__main__":
    sys.exit(main())
