from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from config.settings import KST, settings
from providers.kis_provider import KISProvider


def _write_github_output(values: dict[str, str]) -> None:
    out_path = os.getenv("GITHUB_OUTPUT", "").strip()
    if not out_path:
        return
    with Path(out_path).open("a", encoding="utf-8") as f:
        for k, v in values.items():
            f.write(f"{k}={v}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KIS 접근토큰 1일 1회 사전 확보 및 캐시 메타 출력")
    parser.add_argument("--min-ttl-hours", type=float, default=4.5, help="남은 유효시간 최소 기준(시간)")
    parser.add_argument("--skip-weekends", action="store_true", help="주말(토·일)에는 신규 토큰 발급 스킵")
    parser.add_argument("--github-output", action="store_true", help="GITHUB_OUTPUT에 캐시 저장 여부 기록")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if settings.data_provider != "kis" or not settings.kis_app_key or not settings.kis_app_secret:
        print(f"[KIS TOKEN] data_provider={settings.data_provider} — 토큰 사전 확보 스킵")
        if args.github_output:
            _write_github_output({"needs_save": "false", "cache_key": ""})
        return 0

    now_kst = datetime.now(tz=KST)
    if args.skip_weekends and now_kst.weekday() >= 5:
        print(f"[KIS TOKEN] 주말({now_kst.strftime('%Y-%m-%d')}) — 토큰 발급 스킵")
        if args.github_output:
            _write_github_output({"needs_save": "false", "cache_key": ""})
        return 0

    provider = KISProvider()
    min_ttl_sec = max(120.0, float(args.min_ttl_hours) * 3600.0)
    res = provider.ensure_daily_token(min_ttl_sec=min_ttl_sec, retry_delay_sec=62.0)

    cache_key = f"kis-token-v1-{now_kst.strftime('%Y%m%d')}-{int(time.time())}"
    source = res["source"]
    rem_h = round(res["remaining_sec"] / 3600.0, 2)
    needs_save = "true" if source == "issued" else "false"
    print(f"[KIS TOKEN] source={source} | remaining={rem_h}h | issued_date_kst={res.get('issued_date_kst')}")

    if args.github_output:
        _write_github_output({"needs_save": needs_save, "cache_key": cache_key})
    return 0


if __name__ == "__main__":
    sys.exit(main())
