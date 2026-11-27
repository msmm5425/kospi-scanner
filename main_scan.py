from __future__ import annotations

import argparse
import logging
import sys

from config.settings import settings
from engines.scanner import run_scan
from notify.telegram_bot import TelegramNotifier
from providers.factory import create_provider


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KOSPI 1:3 스윙 스캐너 장중 스캔 실행")
    parser.add_argument("--provider", choices=["pykrx", "kis", "kb"], default=None)
    parser.add_argument("--date", default=None, help="기준 날짜 YYYY-MM-DD (과거 시뮬레이션용)")
    parser.add_argument("--dry-run", action="store_true", help="저장/실제 텔레그램 발송 없이 시뮬레이션")
    parser.add_argument("--no-notify", action="store_true", help="텔레그램 발송 생략")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    provider = create_provider(provider_name=args.provider, as_of_date=args.date)
    notifier = TelegramNotifier(dry_run=args.dry_run or settings.dry_run)

    res = run_scan(
        provider=provider,
        notifier=notifier,
        dry_run=args.dry_run,
        send_notifications=not args.no_notify,
    )

    st = res["stats"]
    print(
        f"[SCAN 완료] date={res['signal_date']} | provider={res['provider']} | "
        f"universe={st['universe_count']} -> matched={st['matched_count']} "
        f"(new={st['new_signals_saved']}, rr_rejected={st['rr_rejected']}) | "
        f"{res['elapsed_sec']}s"
    )
    for sig in res["signals"]:
        rr = sig["rr_geometry"]
        print(
            f"  - {sig['signal_id']} | {sig['name']}({sig['code']}) | {sig['track']} | "
            f"진입 {sig['entry_price']:,}원 | 1R -{rr['risk_pct']:.2f}% | "
            f"여력 +{rr['clear_upside_pct']:.1f}% | R:R 1:{rr['projected_rr']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
