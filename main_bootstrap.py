from __future__ import annotations

import argparse
import logging
import sys

from config.settings import settings
from engines.bootstrap_learner import run_bootstrap_learning
from providers.factory import create_provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="과거 거래일 워크포워드 부트스트랩 자동 학습 실행")
    parser.add_argument("--provider", choices=["pykrx", "kis", "kb"], default=None)
    parser.add_argument("--days", type=int, default=25, help="워크포워드 리플레이 최근 거래일 수 (기본 25일)")
    parser.add_argument("--max-stocks", type=int, default=60, help="상위 유동성 종목 조회 상한 (기본 60종목)")
    parser.add_argument("--dry-run", action="store_true", help="저장 없이 드라이런")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    provider = create_provider(provider_name=args.provider)
    res = run_bootstrap_learning(
        provider=provider,
        lookback_days=args.days,
        max_stocks=args.max_stocks,
        dry_run=args.dry_run,
    )
    learn = res.get("learning") or {}
    sc = learn.get("sample_counts") or {}
    print(
        f"[BOOTSTRAP 완료] days={res.get('days_replayed', 0)} | stocks={res.get('stocks_loaded', 0)} | "
        f"signals_created={res.get('signals_created', 0)} | completed={sc.get('effective_completed', 0)} | "
        f"phase4={((learn.get('phase4_policy') or {}).get('phase4_active'))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
