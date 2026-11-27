from __future__ import annotations

import argparse
import logging
import sys

from datetime import datetime

from config.settings import KST, settings
from engines.auto_learner import run_auto_learning
from engines.bootstrap_learner import run_bootstrap_learning
from engines.error_log import update_error_logs
from engines.top20_analyzer import analyze_top20
from notify.telegram_bot import TelegramNotifier
from providers.factory import create_provider
from storage.repo import repo as default_repo


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="15:50 오답노트 갱신 + 코스피 TOP 20 역추적 + 자동학습")
    parser.add_argument("--provider", choices=["pykrx", "kis", "kb"], default=None)
    parser.add_argument("--date", default=None, help="분석 기준 날짜 YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="저장/실제 텔레그램 발송 없이 실행")
    parser.add_argument("--no-notify", action="store_true", help="텔레그램 발송 생략")
    parser.add_argument("--force", action="store_true", help="공휴일/비개장일 가드 무시 강제 실행")
    parser.add_argument("--bootstrap-if-empty", action="store_true", help="누적 표본이 0건일 때 최근 20일 자동 부트스트랩")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    provider = create_provider(provider_name=args.provider, as_of_date=args.date)
    notifier = TelegramNotifier(dry_run=args.dry_run or settings.dry_run)

    # 평일 한국 공휴일 등 KRX 비개장일 중복 분석·알림 방지
    today_kst = datetime.now(tz=KST).strftime("%Y-%m-%d")
    if args.date is None and not args.force:
        idx_df = provider.get_index_ohlcv(bars=5)
        latest_idx_date = str(idx_df["date"].iloc[-1]) if not idx_df.empty else ""
        if latest_idx_date != today_kst:
            print(f"[ANALYZE 스킵] 오늘은 KRX 휴장일/비개장일입니다 (오늘={today_kst}, 최근 거래일={latest_idx_date}).")
            return 0

    has_bootstrapped = bool(default_repo.get_signals()) or bool(default_repo.load_top20_doc().get("history"))
    if args.bootstrap_if_empty and not has_bootstrapped:
        boot_res = run_bootstrap_learning(
            provider=provider,
            storage=default_repo,
            lookback_days=20,
            max_stocks=60,
            dry_run=args.dry_run,
        )
        print(
            f"[AUTO BOOTSTRAP] 초기 표본 자동 적재: signals={boot_res.get('signals_created', 0)}, "
            f"completed={boot_res.get('error_completed', 0)}"
        )

    err_res = update_error_logs(provider=provider, as_of_date=args.date, dry_run=args.dry_run)
    top20_res = analyze_top20(provider=provider, as_of_date=args.date, dry_run=args.dry_run)
    learn_res = run_auto_learning(
        as_of_date=top20_res["summary"]["date"],
        dry_run=args.dry_run,
    )

    if not args.no_notify:
        notifier.send_daily_analysis(err_res, top20_res, learning_result=learn_res)

    sm = top20_res["summary"]
    sc = learn_res["sample_counts"]
    print(
        f"[ANALYZE 완료] date={sm['date']} | error_updated={err_res['updated']} "
        f"(completed={err_res['completed']}) | TOP20 Hit={sm['hit_count']}/{sm['top20_count']} "
        f"({sm['hit_rate_pct']:.1f}%) | pattern_added={top20_res['pattern_entries_added']} | "
        f"learned_completed={sc['effective_completed']} phase4={learn_res['phase4_policy']['phase4_active']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
