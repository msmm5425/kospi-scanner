# KOSPI 1:3 Swing Scanner (`kospi-swing-scanner`)

단기(1~7거래일) **+10~30% 수익 · 손익비 1:3(R:R >= 3.0)**을 목표로 설계된 **코스피 보통주 전용 어댑터형 스윙 스캐너**입니다.

## 핵심 특징
- **Adapter 아키텍처 (`providers/`)**: 증권사 API 발급 전에는 `DATA_PROVIDER=pykrx` 무료 데이터(`pykrx` + `FinanceDataReader`)로 즉시 실행·검증하고, API 발급 후 `.env` 1줄 변경(`kis` 또는 `kb`)으로 전환합니다.
- **1:3 손익비 기하학 검증 (`risk/rr_geometry.py`)**: 트랙별 무효화 기준점(`1R`) 허용 상한(-5.0% ~ -6.5%)과 상단 거래량 2배 이상 매물대 공백(`>= +10%`, 120일 신고가 블루스카이 `+30%`)을 계산해 `projected_rr >= 3.0` 타점만 선별합니다.
- **3대 정량 전략 (`strategies/`)**:
  - `Track A`: VCP 타이트 변동성 축소 돌파 (Minervini)
  - `Track B`: 기준봉(+9.5%, 250% 거래량) 눌림목 재점화 (Qullamaggie / O'Neil)
  - `Track C`: 바닥권 OBV 선행 돌파 + 연속 기관·외인 수급 매집 전환
- **오답노트 & 15:50 TOP 20 역추적 (`engines/`)**: `D+1~D+10` MFE/MAE/R배수 자동 채점 및 5종 오답 태깅, 15:50 상승률 TOP 20 Hit/Miss 역산, 6종 신규 패턴(Track D 후보) 큐 축적.
- **Phase 1~3 무고정 SL/TP 실측 원칙**: 포착 알림에서는 무효화 참고선(`1R`)과 매물대 공백만 안내하고, 금요일 주간 리포트에서 `3R/+10%` 달성 종목의 MAE 85분위수를 누적해 Phase 4에서 SL/TP를 최종 확정합니다.

## 빠른 시작
```bash
pip install -r requirements.txt
cp .env.example .env

# 단위 테스트
pytest -q

# 과거 특정 날짜 드라이런 시뮬레이션
python -m tests.test_replay --date 2026-11-20

# 장중 스캔 / 장 마감 분석 / 주간 리포트
python main_scan.py --dry-run
python main_analyze.py --dry-run
python main_weekly.py --dry-run
```

## GitHub Actions Secrets 설정
리포지토리 `Settings -> Secrets and variables -> Actions`에 아래 값을 등록하면 `.github/workflows/`의 평일 장중/마감/주간 스케줄이 자동 실행됩니다.
- `DATA_PROVIDER` (`pykrx` 또는 `kis`)
- `KIS_APP_KEY`, `KIS_APP_SECRET` (한투 연동 시)
- `KOSPI_TELEGRAM_TOKEN`, `KOSPI_TELEGRAM_CHAT_ID`
