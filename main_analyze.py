from __future__ import annotations

import argparse
import logging
import sys

from config.settings import settings
from engines.error_log import update_error_logs
from engines.top20_analyzer import analyze_top20
from notify.telegram_bot import TelegramNotifier
from providers.factory import create_provider


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="15:50 오답노트 갱신 + 코스피 TOP 20 역추적")
    parser.add_argument("--provider", choices=["pykrx", "kis", "kb"], default=None)
    parser.add_argument("--date", default=None, help="분석 기준 날짜 YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="저장/실제 텔레그램 발송 없이 실행")
    parser.add_argument("--no-notify", action="store_true", help="텔레그램 발송 생략")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    provider = create_provider(provider_name=args.provider, as_of_date=args.date)
    notifier = TelegramNotifier(dry_run=args.dry_run or settings.dry_run)

    err_res = update_error_logs(provider=provider, as_of_date=args.date, dry_run=args.dry_run)
    top20_res = analyze_top20(provider=provider, as_of_date=args.date, dry_run=args.dry_run)

    if not args.no_notify:
        notifier.send_daily_analysis(err_res, top20_res)

    sm = top20_res["summary"]
    print(
        f"[ANALYZE 완료] date={sm['date']} | error_updated={err_res['updated']} "
        f"(completed={err_res['completed']}) | TOP20 Hit={sm['hit_count']}/{sm['top20_count']} "
        f"({sm['hit_rate_pct']:.1f}%) | pattern_added={top20_res['pattern_entries_added']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
