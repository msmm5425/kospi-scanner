from __future__ import annotations

import argparse
import logging
import sys

from config.settings import settings
from engines.weekly_report import build_weekly_report
from notify.telegram_bot import TelegramNotifier, format_weekly_report_message


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="금요일 16:30 코스피 주간 스윙 성과 리포트")
    parser.add_argument("--date", default=None, help="기준 날짜 YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="텔레그램 발송 없이 콘솔 출력")
    parser.add_argument("--no-notify", action="store_true", help="텔레그램 발송 생략")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    report = build_weekly_report(as_of_date=args.date)
    msg = format_weekly_report_message(report)
    print(msg)

    if not args.no_notify:
        notifier = TelegramNotifier(dry_run=args.dry_run or settings.dry_run)
        notifier.send_text(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
