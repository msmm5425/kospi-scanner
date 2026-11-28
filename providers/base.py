from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from config.settings import settings

_EXCLUDED_SUFFIX_RE = re.compile(r"(?:스팩\d*호|\d+호스팩|\d+우[A-Z]?|우선주|REITs?)$", re.IGNORECASE)
_ETF_BRAND_PREFIXES: tuple[str, ...] = (
    "KODEX ", "TIGER ", "SOL ", "ACE ", "RISE ", "PLUS ", "KBSTAR ", "HANARO ",
    "ARIRANG ", "KOSEF ", "KIWOOM ", "TIMEFOLIO ", "KOACT ", "1Q ", "BNK ",
    "HK ", "WON ", "VITA ", "FOCUS ", "TREX ", "SMART ", "MASTER ", "UNICORN ",
    "TIME ", "TRUSTON ",
)


def is_eligible_common_stock(code: str, name: str = "") -> bool:
    """코스피 6자리 끝자리 '0' 순수 보통주 여부 및 스팩·리츠·ETF·우선주 제외 검증."""
    c = str(code or "").strip()
    if len(c) != 6 or not c.isdigit() or not c.endswith("0"):
        return False

    n = str(name or "").strip()
    if not n:
        return True

    upper_n = n.upper()
    if upper_n.startswith(_ETF_BRAND_PREFIXES):
        return False
    for kw in settings.excluded_name_keywords:
        if kw.upper() in upper_n:
            return False

    if _EXCLUDED_SUFFIX_RE.search(n):
        return False
    return True


class MarketDataProvider(ABC):
    """전략·분석 엔진이 사용하는 표준 6종 시장 데이터 인터페이스."""

    provider_name: str = "base"

    @abstractmethod
    def get_universe(
        self,
        min_cap: int | None = None,
        max_cap: int | None = None,
    ) -> list[dict[str, Any]]:
        """코스피 보통주 필터 및 시총 범위 적용 유니버스 반환.

        Returns:
            list[dict]: 각 원소는 {'code', 'name', 'marcap', 'trading_val', 'price', 'chg_pct'} 포함.
        """

    @abstractmethod
    def get_safety_status(self, code: str) -> dict[str, Any]:
        """정리매매·관리종목·거래정지·투자주의/경고/위험 판별 및 현재 시세 반환.

        Returns:
            dict: {'is_safe': bool, 'reject_reason': str, 'price': float,
                   'chg_pct': float, 'trading_val': float}
        """

    @abstractmethod
    def get_daily_ohlcv(
        self,
        code: str,
        bars: int = 150,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """수정주가 적용 일봉 반환.

        Returns:
            DataFrame columns: ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
            ascending by date ('YYYY-MM-DD').
        """

    @abstractmethod
    def get_investor_supply(
        self,
        code: str,
        days: int = 5,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """최근 N거래일 외인·기관 순매수량(주) 집계 반환.

        Returns:
            dict: {'frgn_today': int, 'orgn_today': int, 'frgn_3d': int, 'orgn_3d': int,
                   'total_5d': int, 'pos_days_5d': int, 'intraday_pending': bool}
        """

    @abstractmethod
    def get_index_ohlcv(
        self,
        bars: int = 100,
        end_date: str | None = None,
        market: str = "KOSPI",
    ) -> pd.DataFrame:
        """KOSPI(KS11) 또는 KOSDAQ(KQ11) 지수 일봉 반환.

        Returns:
            DataFrame columns: ['date', 'close', 'chg_pct'] ascending by date.
        """

    @abstractmethod
    def get_top_gainers(
        self,
        top_n: int = 20,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """보통주 상승률 상위 top_n 종목 반환.

        Returns:
            list[dict]: {'code', 'name', 'close', 'chg_pct', 'trading_val', 'marcap'}
        """

    def get_sector_map(self) -> dict[str, str]:
        """선택적 업종 매핑 {code: sector} 반환 (미지원 프로바이더는 빈 dict)."""
        return {}

    def is_trading_day(self, date_str: str | None = None) -> bool:
        """해당 날짜가 KRX 주식시장 개장일인지 여부 반환 (주말 제외 기본 True)."""
        from datetime import datetime
        from config.settings import KST

        eff = date_str or datetime.now(tz=KST).strftime("%Y-%m-%d")
        try:
            return datetime.strptime(eff[:10], "%Y-%m-%d").weekday() < 5
        except ValueError:
            return True

    def get_session_quote(self, code: str, session: str = "REGULAR") -> dict[str, Any] | None:
        """프리장/애프터장(통합 UN) 또는 정규장(J) 참고 시세 조회 (미지원 시 None)."""
        return None


def get_market_index_ohlcv(
    provider: MarketDataProvider,
    bars: int = 100,
    end_date: str | None = None,
    market: str = "KOSPI",
) -> pd.DataFrame:
    """하위호환 프로바이더/테스트 목을 보존하며 시장별(KOSPI/KOSDAQ) 지수 일봉을 안전 조회한다."""
    mkt = str(market or "KOSPI").upper()
    if mkt != "KOSPI":
        try:
            return provider.get_index_ohlcv(bars=bars, end_date=end_date, market=mkt)
        except TypeError:
            pass
    return provider.get_index_ohlcv(bars=bars, end_date=end_date)
