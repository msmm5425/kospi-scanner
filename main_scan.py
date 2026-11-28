from __future__ import annotations

import argparse
import logging
import sys

from datetime import datetime

from config.settings import KST, settings
from engines.closing_picker import build_and_send_closing_briefing
from engines.scanner import run_loop_mode, run_scan
from notify.telegram_bot import TelegramNotifier, format_scan_diagnostic_summary
from providers.factory import create_provider


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KOSPI 1:3 스윙 스캐너 08:00~21:00 스캔 실행")
    parser.add_argument("--provider", choices=["pykrx", "kis", "kb"], default=None)
    parser.add_argument("--date", default=None, help="기준 날짜 YYYY-MM-DD (과거 시뮬레이션용)")
    parser.add_argument("--latest-trading-day", action="store_true", help="휴장일에도 직전 유효 거래일 차트 기준으로 스캔")
    parser.add_argument("--dry-run", action="store_true", help="저장/실제 텔레그램 발송 없이 시뮬레이션")
    parser.add_argument("--no-notify", action="store_true", help="텔레그램 발송 생략")
    parser.add_argument("--force", action="store_true", help="휴장일/비개장일 가드 무시 강제 실행")
    parser.add_argument("--loop", action="store_true", help="지정 시간 동안 연속 루프 스캔 실행")
    parser.add_argument("--interval-sec", type=int, default=settings.loop_interval_sec, help="루프 스캔 주기(초)")
    parser.add_argument("--max-minutes", type=int, default=260, help="루프 세션 최대 실행 시간(분)")
    parser.add_argument("--end-kst", default="21:05", help="루프 자동 종료 KST 시각 (HH:MM)")
    parser.add_argument("--git-sync", action="store_true", help="루프 중 신규 포착 시 Git 자동 커밋/푸시")
    parser.add_argument("--closing-briefing", action="store_true", help="당일 포착 종목 15:00 종가 매수 재검증 브리핑 실행")
    parser.add_argument("--notify-summary", action="store_true", help="단일 스캔 진단 요약(0건/근접탈락 포함) 출력 및 텔레그램 전송")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    try:
        return _run_main(args)
    except Exception:
        logging.getLogger(__name__).exception("스캔 워크플로 치명적 예외 발생")
        raise


def _run_main(args: argparse.Namespace) -> int:
    target_date = args.date
    provider = create_provider(provider_name=args.provider, as_of_date=target_date)
    notifier = TelegramNotifier(dry_run=args.dry_run or settings.dry_run)

    today_kst = datetime.now(tz=KST).strftime("%Y-%m-%d")
    if target_date is None and args.latest_trading_day:
        idx_df = provider.get_index_ohlcv(bars=10)
        if not idx_df.empty:
            target_date = str(idx_df["date"].iloc[-1])
            provider.as_of_date = target_date

    if target_date is None and not args.force:
        if not provider.is_trading_day(today_kst):
            print(f"[SCAN 스킵] {today_kst}은(는) KRX 휴장일/비개장일입니다.")
            return 0

    scan_now_kst = (
        datetime.strptime(target_date[:10], "%Y-%m-%d").replace(hour=15, minute=35, tzinfo=KST)
        if target_date
        else None
    )

    if args.closing_briefing:
        brief = build_and_send_closing_briefing(
            provider=provider,
            notifier=notifier,
            now_kst=scan_now_kst,
            dry_run=args.dry_run,
            send_notifications=not args.no_notify,
            force=args.force,
        )
        print(
            f"[15:00 종가 브리핑] date={brief.get('date')} | qualified={len(brief.get('qualified') or [])} | "
            f"invalidated={len(brief.get('invalidated') or [])}"
        )
        return 0

    if args.loop and target_date is None:
        loop_res = run_loop_mode(
            provider=provider,
            notifier=notifier,
            interval_sec=args.interval_sec,
            max_minutes=args.max_minutes,
            end_kst=args.end_kst,
            dry_run=args.dry_run,
            send_notifications=not args.no_notify,
            git_sync=args.git_sync,
            force=args.force,
        )
        print(
            f"[LOOP SUMMARY] cycles={loop_res['cycles']} | new_signals={loop_res['total_new_signals']}"
        )
        return 0

    res = run_scan(
        provider=provider,
        notifier=notifier,
        now_kst=scan_now_kst,
        dry_run=args.dry_run,
        send_notifications=not args.no_notify,
    )

    st = res["stats"]
    print(
        f"[SCAN 완료] date={res['signal_date']} ({res['session']}) | provider={res['provider']} | "
        f"universe={st['universe_count']} -> matched={st['matched_count']} "
        f"(new={st['new_signals_saved']}, rr_rejected={st['rr_rejected']}) | "
        f"{res['elapsed_sec']}s"
    )
    for sig in res["signals"]:
        rr = sig["rr_geometry"]
        print(
            f"  - {sig['signal_id']} | {sig['name']}({sig['code']}) [{sig.get('session')}] | {sig['track']} | "
            f"정규장 {sig['entry_price']:,}~{sig.get('max_entry_price', sig['entry_price']):,}원 | "
            f"1R -{rr['risk_pct']:.2f}% | 여력 +{rr['clear_upside_pct']:.1f}% | R:R 1:{rr['projected_rr']:.2f}"
        )

    if args.notify_summary or args.verbose:
        summary_msg = format_scan_diagnostic_summary(res)
        print("\n" + summary_msg)
        if args.notify_summary and not args.no_notify:
            summary_bot = TelegramNotifier(dry_run=settings.dry_run)
            summary_bot.send_text(summary_msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
