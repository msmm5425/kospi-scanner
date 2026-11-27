from __future__ import annotations

from config.settings import settings
from .base import MarketDataProvider
from .kb_provider import KBProvider
from .kis_provider import KISProvider
from .pykrx_provider import PyKRXProvider


def create_provider(
    provider_name: str | None = None,
    as_of_date: str | None = None,
) -> MarketDataProvider:
    """DATA_PROVIDER 설정값에 따라 적절한 MarketDataProvider 인스턴스를 반환."""
    name = (provider_name or settings.data_provider or "pykrx").strip().lower()
    if name == "pykrx":
        return PyKRXProvider(as_of_date=as_of_date)
    if name == "kis":
        return KISProvider(as_of_date=as_of_date)
    if name == "kb":
        return KBProvider(as_of_date=as_of_date)
    raise ValueError(
        f"지원하지 않는 DATA_PROVIDER='{name}' 입니다. ('pykrx', 'kis', 'kb' 중 선택)"
    )
