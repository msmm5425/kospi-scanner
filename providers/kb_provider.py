from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
import requests

from config.settings import settings
from .base import MarketDataProvider

logger = logging.getLogger(__name__)


class KBProvider(MarketDataProvider):
    """[연동 대기] KB증권 Open API 어댑터 (국내 IP / Self-hosted Runner 환경 전용)."""

    provider_name = "kb"

    def __init__(self, as_of_date: str | None = None):
        self.as_of_date = as_of_date
        self.app_key = settings.kb_app_key
        self.app_secret = settings.kb_app_secret
        self.base_url = settings.kb_api_base_url.rstrip("/")
        self._session = requests.Session()
        self._token = ""

    def _guard_environment(self) -> None:
        if os.getenv("GITHUB_ACTIONS", "").lower() == "true" and not os.getenv("SELF_HOSTED_KR"):
            logger.warning(
                "KB증권 Open API는 해외 IP 차단 정책으로 인해 일반 GitHub Actions 러너에서는 "
                "차단될 수 있습니다. 로컬 PC / 국내 리전 / Self-hosted Runner를 사용하세요."
            )
        if not self.app_key or not self.app_secret or not self.base_url:
            raise RuntimeError(
                "KB증권 API 설정(KB_APP_KEY, KB_APP_SECRET, KB_API_BASE_URL)이 필요합니다. "
                "국내 IP 환경에서 키 발급 후 .env를 설정하세요."
            )

    def get_universe(
        self,
        min_cap: int | None = None,
        max_cap: int | None = None,
    ) -> list[dict[str, Any]]:
        self._guard_environment()
        raise NotImplementedError("KB증권 Open API 유니버스 엔드포인트 연결 대기 중")

    def get_safety_status(self, code: str) -> dict[str, Any]:
        self._guard_environment()
        raise NotImplementedError("KB증권 Open API 현재가/종목상태 엔드포인트 연결 대기 중")

    def get_daily_ohlcv(
        self,
        code: str,
        bars: int = 150,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        self._guard_environment()
        raise NotImplementedError("KB증권 Open API 일봉 조회 엔드포인트 연결 대기 중")

    def get_investor_supply(
        self,
        code: str,
        days: int = 5,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        self._guard_environment()
        raise NotImplementedError("KB증권 Open API 투자자별 수급 엔드포인트 연결 대기 중")

    def get_index_ohlcv(
        self,
        bars: int = 100,
        end_date: str | None = None,
        market: str = "KOSPI",
    ) -> pd.DataFrame:
        self._guard_environment()
        raise NotImplementedError("KB증권 Open API 지수 일봉 엔드포인트 연결 대기 중")

    def get_top_gainers(
        self,
        top_n: int = 20,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        self._guard_environment()
        raise NotImplementedError("KB증권 Open API 상승률 순위 엔드포인트 연결 대기 중")
