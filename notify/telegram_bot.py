from __future__ import annotations

import logging
import time
from typing import Any

import requests

from config.settings import settings

logger = logging.getLogger(__name__)


def format_signal_message(signal: dict[str, Any]) -> str:
    """설계도 7장 표준 포맷 (SL/TP 최종 확정 전 1~3단계 실측 포맷) 메시지 생성."""
    name = str(signal.get("name") or "")
    code = str(signal.get("code") or "").zfill(6)
    track_label = str(signal.get("track_label") or signal.get("track") or "")
    entry_p = int(round(float(signal.get("entry_price") or 0)))

    snap = signal.get("snapshot") or {}
    rr = signal.get("rr_geometry") or {}

    regime = str(snap.get("market_regime") or "NORMAL").upper()
    regime_badge = "약세/주의" if regime == "WEAK" else "정상/강세"
    weak_banner = "⚠️ [지수 약세 주의]\n" if regime == "WEAK" else ""

    chg_pct = float(snap.get("chg_pct") or 0.0)
    marcap_uk = float(snap.get("marcap_uk") or 0.0)
    vol_ratio = float(snap.get("vol_ratio_20d") or 0.0)
    rs_20 = float(snap.get("rs_20d_diff") or 0.0)
    frgn_3d = int(snap.get("frgn_3d") or 0)
    orgn_3d = int(snap.get("orgn_3d") or 0)
    obv_trend = str(snap.get("obv_trend") or "우상향")
    atr14_pct = float(snap.get("atr14_pct") or 0.0)

    inval_p = int(round(float(rr.get("invalidation_ref_price") or 0)))
    risk_pct = float(rr.get("risk_pct") or 0.0)
    clear_up = float(rr.get("clear_upside_pct") or 0.0)
    proj_rr = float(rr.get("projected_rr") or 0.0)
    nearest_res = rr.get("nearest_resistance_price")
    is_blue_sky = bool(rr.get("is_blue_sky", False))

    if is_blue_sky or nearest_res is None:
        upside_line = f"120일 신고가 블루스카이 (+{clear_up:.1f}%)"
    else:
        upside_line = f"{int(round(float(nearest_res))):,}원 (+{clear_up:.1f}%)"

    p4 = signal.get("phase4_sl_tp")
    sess_info = signal.get("session_info") or {}
    mkt = str(signal.get("market") or snap.get("market") or "KOSPI").upper()
    session_type = str(signal.get("session") or sess_info.get("session") or "REGULAR").upper()
    max_entry_p = int(round(float(rr.get("max_entry_price") or signal.get("max_entry_price") or entry_p)))

    if session_type == "PREMARKET":
        header_badge = f"⏰ [프리장 포착 · {mkt}] {name} ({code})"
        session_line = f"시장: {mkt} · 프리장(08:00~09:00) 직전 정규장 검증"
    elif session_type == "AFTERMARKET":
        header_badge = f"🌙 [애프터장 포착 · {mkt}] {name} ({code})"
        session_line = f"시장: {mkt} · 장후/애프터장(15:30~21:00) 정규장 종가 검증"
    else:
        header_badge = f"🚨 [{mkt} 스윙 포착] {name} ({code})"
        session_line = f"시장: {mkt} | 장세: {regime_badge}"

    ext_line = ""
    if session_type in {"PREMARKET", "AFTERMARKET"}:
        ext_p = sess_info.get("ext_ref_price")
        ext_gap = sess_info.get("ext_gap_vs_regular_pct")
        ext_mkt = sess_info.get("ext_market_code") or "통합"
        if ext_p and int(ext_p) > 0:
            ext_line = (
                f"▪ 장외 참고가({ext_mkt}): {int(ext_p):,}원 "
                f"(정규장 대비 {float(ext_gap or 0.0):+.2f}% | 유동성 주의)\n"
            )

    entry_band_line = (
        f"• 정규장 유효 진입 구간: {entry_p:,}원 ~ {max_entry_p:,}원 "
        f"({max_entry_p:,}원 초과 갭 시 패스)\n"
        if max_entry_p >= entry_p
        else f"• 진입 기준가: {entry_p:,}원\n"
    )

    if p4 and p4.get("sl_price"):
        tp_parts = " / ".join(
            f"{t['price']:,}원(+{t['pct']:.0f}%)" for t in (p4.get("tp_levels") or [])
        )
        rr_section = (
            f"🎯 손익비 · 실측 확정 SL/TP (Phase 4 · 정규장 기준)\n"
            f"{entry_band_line}"
            f"• 확정 손절: {int(p4['sl_price']):,}원 ({p4['sl_pct']:+.2f}%) | 분할 익절: {tp_parts}\n"
            f"• 1차 도달 시 본절 보호 | 예상 R:R = 1 : {proj_rr:.2f} | ATR(14): {atr14_pct:.2f}%"
        )
    else:
        rr_section = (
            f"📐 정규장 기준 진입 타점 · 손익비 가이드 (1~3단계 실측)\n"
            f"{entry_band_line}"
            f"• 구조적 무효화 기준선(1R): {inval_p:,}원 (-{risk_pct:.2f}%)\n"
            f"• 상단 매물대 공백: {upside_line} | 예상 R:R = 1 : {proj_rr:.2f}\n"
            f"• 14일 ATR: {atr14_pct:.2f}%"
        )

    return (
        f"{weak_banner}"
        f"{header_badge}\n"
        f"▪ 트랙: {track_label} | {session_line}\n"
        f"▪ 정규장 기준가: {entry_p:,}원 ({chg_pct:+.1f}%) | 시총: {marcap_uk:,.0f}억\n"
        f"{ext_line}"
        f"▪ 거래량: 20일 평균 대비 {vol_ratio:.0f}% | RS(20일): 지수 대비 {rs_20:+.1f}%\n"
        f"▪ 3일 수급: 외인 {frgn_3d:+,}주 / 기관 {orgn_3d:+,}주 (OBV {obv_trend})\n\n"
        f"{rr_section}"
    )


def format_top20_analysis_message(
    error_result: dict[str, Any],
    top20_result: dict[str, Any],
    learning_result: dict[str, Any] | None = None,
) -> str:
    """15:50 장 마감 오답노트 + TOP 20 역추적 + 자동학습 진척 브리핑 포맷터."""
    summary = top20_result.get("summary") or {}
    dt = str(summary.get("date") or "")
    hits = summary.get("hits") or []
    misses = summary.get("misses") or []
    has_kq = any(str(x.get("market") or "").upper() == "KOSDAQ" for x in [*hits, *misses])
    mkt_title = "KOSPI·KOSDAQ" if has_kq else "KOSPI"

    lines = [
        f"📊 [{mkt_title} 15:50 장 마감 오답노트 & TOP 20 역추적] ({dt})",
        f"• 오답노트 갱신: 진행 {error_result.get('updated', 0)}건 / D+10 완료 {error_result.get('completed', 0)}건",
        f"• 상승률 TOP 20 선행 포착률: {summary.get('hit_count', 0)}/{summary.get('top20_count', 0)} "
        f"({summary.get('hit_rate_pct', 0.0):.1f}%)",
    ]
    if hits:
        hit_str = ", ".join(
            f"{'[' + h['market'] + '] ' if has_kq and h.get('market') else ''}{h['name']}({h['chg_pct']:+.1f}%)"
            for h in hits[:5]
        )
        lines.append(f"✅ Hit: {hit_str}")
    if misses:
        lines.append("🔍 Miss 병목 역산 TOP 5:")
        for m in misses[:5]:
            mkt_tag = f"[{m['market']}] " if has_kq and m.get("market") else ""
            tags_txt = f" | 전조: {', '.join(m['feature_tags'])}" if m.get("feature_tags") else ""
            lines.append(f"  - {mkt_tag}{m['name']} ({m['chg_pct']:+.1f}%): {m['diagnosis']}{tags_txt}")

    if learning_result:
        sc = learning_result.get("sample_counts") or {}
        p4 = learning_result.get("phase4_policy") or {}
        lines.append("")
        lines.append(
            f"🧠 [자동 학습 ({learning_result.get('learning_days', 0)}일차 · {sc.get('cohort_mode', 'LIVE')})]"
        )
        lines.append(
            f"• 표본: 완료 {sc.get('effective_completed', 0)}/30건 "
            f"(실전 {sc.get('live_completed', 0)}, 승자 {sc.get('effective_winners', 0)}/15) | "
            f"3R 달성률 {learning_result.get('win_3r_rate', 0.0):.1f}% | 평균MFE +{learning_result.get('avg_mfe', 0.0):.1f}%"
        )
        promo = learning_result.get("promoted_track_d_patterns") or []
        if promo:
            lines.append(f"• 활성 Track D: {', '.join(promo)}")
        adjs = learning_result.get("adjustments") or []
        if adjs:
            lines.append("• 오늘의 자동 보정: " + " / ".join(adjs[:3]))
        else:
            lines.append("• 오늘의 자동 보정: 파라미터 유지")
        if learning_result.get("just_activated_phase4"):
            lines.append(
                f"🎓 [Phase 4 실측 SL/TP 자동 전환 완료!] "
                f"SL {p4.get('default_sl_pct', -5.0):+.2f}% | TP {p4.get('tp_levels_pct', [])}"
            )
    return "\n".join(lines)


def format_weekly_report_message(report: dict[str, Any]) -> str:
    """금요일 16:30 주간 리포트 메시지 생성."""
    t_all = (report.get("track_stats") or {}).get("ALL") or {}
    lines = [
        f"📈 [KOSPI 주간 스윙 성과 리포트] ({report.get('as_of_date', '')})",
        f"• 이번 주 포착: {report.get('weekly_signal_count', 0)}건 | 누적: {report.get('total_signal_count', 0)}건",
        f"• 전체 완료 승률(+10%/3R): +10% {t_all.get('hit_10_rate', 0.0):.1f}% | "
        f"3R {t_all.get('hit_3r_rate', 0.0):.1f}% | 평균 MFE {t_all.get('avg_mfe', 0.0):+.1f}%",
        "",
        "▪ 트랙별 요약:",
    ]
    for tr, label in (
        ("TRACK_A_VCP", "Track A"),
        ("TRACK_B_PULLBACK", "Track B"),
        ("TRACK_C_TURNAROUND", "Track C"),
        ("TRACK_D", "Track D(자동)"),
    ):
        st = (report.get("track_stats") or {}).get(tr) or {}
        lines.append(
            f"  - {label}: {st.get('total', 0)}건 (완료 {st.get('completed', 0)}) | "
            f"3R {st.get('hit_3r_rate', 0.0):.1f}% | +10% {st.get('hit_10_rate', 0.0):.1f}%"
        )

    reg = report.get("regime_stats") or {}
    lines.append(
        f"▪ 장세 비교: NORMAL 3R {(reg.get('NORMAL') or {}).get('hit_3r_rate', 0.0):.1f}% vs "
        f"WEAK 3R {(reg.get('WEAK') or {}).get('hit_3r_rate', 0.0):.1f}%"
    )

    errs = report.get("top3_errors") or []
    if errs:
        lines.append("▪ 오답 TOP 3: " + ", ".join(f"{e['tag']}({e['count']}건)" for e in errs))

    d_cands = report.get("track_d_candidates") or []
    if d_cands:
        lines.append("▪ 최근 4주 Track D 신규 전략 후보:")
        for c in d_cands[:3]:
            lines.append(
                f"  - {c['pattern']}: {c['count']}회 (1:3 R:R 가능 {c['rr_viable_count']}회, {c['rr_viable_rate_pct']:.0f}%)"
            )

    p4 = report.get("phase4_distribution") or {}
    sl_rec = p4.get("recommended_sl_mae_85pct")
    mfe_p = p4.get("mfe_percentiles") or {}
    status_txt = "Phase 4 전환 가능(표본 30+)" if report.get("phase4_ready") else f"실측 누적 중(성공 표본 {report.get('winner_sample_count', 0)}/30)"
    lines.append(f"▪ SL/TP 분포 ({status_txt}):")
    if sl_rec is not None and mfe_p:
        lines.append(
            f"  - 승리종목 MAE 85분위 SL 참고: {sl_rec:+.2f}% | "
            f"MFE P25/P50/P75: +{mfe_p.get('p25', 0):.1f}% / +{mfe_p.get('p50', 0):.1f}% / +{mfe_p.get('p75', 0):.1f}%"
        )
    else:
        lines.append("  - 완료 승자 표본 축적 대기 중")

    lp = report.get("learning_profile") or {}
    ap = lp.get("adaptive_params") or {}
    promo_d = lp.get("promoted_track_d_patterns") or []
    if ap:
        lines.append(
            f"▪ 자동 보정 파라미터 ({lp.get('learning_days', 0)}일차): "
            f"윗꼬리<={ap.get('max_upper_wick_ratio', settings.max_upper_wick_ratio):.2f} | "
            f"최소R:R 1:{ap.get('min_projected_rr', settings.min_projected_rr):.1f} | "
            f"약세R:R 1:{ap.get('weak_regime_min_rr', settings.weak_regime_min_rr):.1f}"
        )
    if promo_d:
        lines.append(f"▪ 실전 투입 중 Track D: {', '.join(promo_d)}")

    return "\n".join(lines)


def format_closing_briefing_message(briefing: dict[str, Any]) -> str:
    """15:00 KST 장 마감 전 최종 매수 픽 브리핑 메시지 생성."""
    dt = str(briefing.get("date") or "")
    checked = str(briefing.get("checked_at") or "15:00 KST")
    regime = "약세/주의 ⚠️" if str(briefing.get("market_regime", "NORMAL")).upper() == "WEAK" else "정상/강세"
    qualified = briefing.get("qualified") or []
    invalidated = briefing.get("invalidated") or []
    total_today = int(briefing.get("total_today_signals", 0))

    lines = [
        f"🏁 [15:00 장 마감 매수 최종 브리핑] ({dt} · {checked})",
        f"▪ 금일 스캔 포착: {total_today}건 → 종가 매수 적격 {len(qualified)}건 / 보류 {len(invalidated)}건",
        f"▪ 시장 장세: {regime}",
    ]

    if qualified:
        medals = ("🥇", "🥈", "🥉")
        lines.append("")
        lines.append("✅ [장 마감 전 최종 매수 픽]")
        for idx, item in enumerate(qualified[:5]):
            badge = medals[idx] if idx < len(medals) else f"#{idx + 1}"
            res_txt = (
                f"블루스카이(+{item['closing_upside_pct']:.1f}%)"
                if item.get("is_blue_sky") or not item.get("nearest_resistance_price")
                else f"{item['nearest_resistance_price']:,}원(+{item['closing_upside_pct']:.1f}%)"
            )
            mkt_badge = f"[{item['market']}] " if item.get("market") == "KOSDAQ" else ""
            lines.append(
                f"{badge} {idx + 1}순위: {mkt_badge}{item['name']} ({item['code']}) | {item['track_label']}"
            )
            lines.append(
                f"  • 15:00 현재가: {item['closing_price']:,}원 ({item['closing_chg_pct']:+.1f}%) | "
                f"윗꼬리 {item['closing_wick']:.2f} | 점수 {item['closing_score']:.1f}"
            )
            lines.append(
                f"  • 1R 방어선: {item['invalidation_ref_price']:,}원 (-{item['closing_risk_pct']:.2f}%) | "
                f"상단 {res_txt} | R:R 1 : {item['closing_rr']:.2f}"
            )
            lines.append(
                f"  • 동시호가 유효 상한: {item['max_entry_price']:,}원 이하 매수 권장"
            )
    else:
        lines.append("")
        lines.append("⏸️ 금일 15:00 기준 1:3 손익비 + 종가 안착 조건을 통과한 최종 매수 적합 종목이 없습니다 (금일 관망).")

    if invalidated:
        lines.append("")
        lines.append("⚠️ [장중 포착 → 15:00 종가 매수 보류 종목]")
        for inv in invalidated[:5]:
            rsn = ", ".join(inv.get("invalidate_reasons") or ["기준 이탈"])
            lines.append(
                f"  - {inv['name']}({inv['code']}) {inv['closing_price']:,}원: {rsn}"
            )

    return "\n".join(lines)


def format_scan_diagnostic_summary(scan_result: dict[str, Any]) -> str:
    """단일 스캔/차트 점검 테스트 결과(포착 픽 + 병목/근접 탈락) 요약 메시지 생성."""
    st = scan_result.get("stats") or {}
    sigs = scan_result.get("signals") or []
    rr_rej = scan_result.get("rr_rejections") or []
    near = scan_result.get("near_misses") or []
    markets = {str(m).upper() for m in (scan_result.get("markets") or [])}
    has_kq = "KOSDAQ" in markets or any(
        str(x.get("market") or "").upper() == "KOSDAQ" for x in [*sigs, *rr_rej, *near]
    )
    mkt_title = "KOSPI·KOSDAQ" if has_kq else "KOSPI"

    err_cnt = int(st.get("inspect_errors", 0) or 0)
    err_samples = scan_result.get("error_samples") or []
    err_suffix = f" / 오류 {err_cnt}건" if err_cnt > 0 else ""

    lines = [
        f"🔎 [{mkt_title} 차트 매수픽 점검] ({scan_result.get('signal_date', '')} · {scan_result.get('session', '')})",
        f"▪ 프로바이더: {scan_result.get('provider', '')} | 소요: {scan_result.get('elapsed_sec', 0)}s",
        f"▪ 유니버스 {st.get('universe_count', 0)}종목 → 사전제외 {st.get('pre_filtered_out', 0)} / "
        f"안전제외 {st.get('safety_rejected', 0)} / 관문탈락 {st.get('gate_rejected', 0)} / "
        f"전략탈락 {st.get('strategy_rejected', 0)} / R:R탈락 {st.get('rr_rejected', 0)}{err_suffix}",
        f"▪ 최종 스윙 매수픽 통과: {len(sigs)}건",
    ]

    if sigs:
        lines.append("")
        lines.append("✅ [최종 통과 매수픽]")
        for s in sigs[:8]:
            rr = s.get("rr_geometry") or {}
            m_tag = f"[{s.get('market', 'KOSPI')}] " if has_kq else ""
            lines.append(
                f"  • {m_tag}{s['name']}({s['code']}) | {s.get('track_label') or s.get('track')}\n"
                f"    기준가 {int(s['entry_price']):,}~{int(s.get('max_entry_price', s['entry_price'])):,}원 | "
                f"1R -{float(rr.get('risk_pct', 0)):.2f}% | 여력 +{float(rr.get('clear_upside_pct', 0)):.1f}% | "
                f"R:R 1:{float(rr.get('projected_rr', 0)):.2f}"
            )
    else:
        lines.append("")
        lines.append(
            f"⏸️ 현재 차트 기준 Track A/B/C/D + 손익비(1:{settings.min_projected_rr:.1f}+) 조건을 모두 통과한 매수픽이 0건입니다."
        )

    if rr_rej:
        lines.append("")
        lines.append("📐 [전략 통과 → R:R 단계 아쉽게 탈락]")
        for r in rr_rej[:5]:
            m_tag = f"[{r.get('market', 'KOSPI')}] " if has_kq else ""
            lines.append(
                f"  - {m_tag}{r['name']}({r['code']}) | {r['track']} | "
                f"R:R 1:{float(r.get('projected_rr', 0)):.2f} ({r.get('reason', '')})"
            )

    if near:
        lines.append("")
        lines.append("👀 [공통관문 통과 · 전략 조건 근접 TOP 5]")
        for m in near[:5]:
            rsn = "; ".join(m.get("reasons") or [])
            m_tag = f"[{m.get('market', 'KOSPI')}] " if has_kq else ""
            lines.append(
                f"  - {m_tag}{m['name']}({m['code']}) {int(m['price']):,}원({float(m['chg_pct']):+.1f}%) | "
                f"{m['track']} {m['passed_count']}/{m['total_checks']} [{rsn}]"
            )

    if err_cnt > 0 and err_samples:
        lines.append("")
        lines.append("⚠️ [점검 중 예외 종목]")
        for sample in err_samples[:5]:
            lines.append(f"  - {sample}")

    return "\n".join(lines)


class TelegramNotifier:
    """주식 전용 텔레그램 전송기 (KOSPI_TELEGRAM_TOKEN / CHAT_ID 분리 사용)."""

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        dry_run: bool | None = None,
    ):
        self.token = (settings.telegram_token if token is None else token).strip()
        self.chat_id = (settings.telegram_chat_id if chat_id is None else chat_id).strip()
        self.dry_run = settings.dry_run if dry_run is None else bool(dry_run)

    @staticmethod
    def _split_chunks(text: str, limit: int = 3900) -> list[str]:
        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        curr: list[str] = []
        curr_len = 0
        for line in text.splitlines(keepends=True):
            if curr_len + len(line) > limit and curr:
                chunks.append("".join(curr).rstrip())
                curr = [line]
                curr_len = len(line)
            else:
                curr.append(line)
                curr_len += len(line)
        if curr:
            chunks.append("".join(curr).rstrip())
        return chunks

    def send_text(self, text: str) -> bool:
        if not text:
            return False
        if self.dry_run or not self.token or not self.chat_id:
            reason = "DRY_RUN" if self.dry_run else "토큰 미설정"
            logger.info("[Telegram %s]\n%s", reason, text)
            print(f"\n[Telegram {reason}]\n{text}\n")
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        ok_all = True
        for chunk in self._split_chunks(text):
            sent = False
            for attempt in range(3):
                try:
                    resp = requests.post(
                        url,
                        json={
                            "chat_id": self.chat_id,
                            "text": chunk,
                            "disable_web_page_preview": True,
                        },
                        timeout=12.0,
                    )
                    if resp.status_code == 200:
                        sent = True
                        time.sleep(0.4)
                        break
                    if resp.status_code == 429:
                        wait_s = int((resp.json().get("parameters") or {}).get("retry_after") or 3)
                        time.sleep(wait_s + 0.5)
                        continue
                    logger.warning("Telegram HTTP %s: %s", resp.status_code, resp.text[:200])
                except Exception as e:
                    logger.warning("Telegram exception (%s/3): %s", attempt + 1, e)
                time.sleep(1.0)
            if not sent:
                ok_all = False
        if not ok_all:
            logger.error("Telegram 메시지 전송에 최종 실패했습니다 (chat_id=%s).", self.chat_id)
            print(f"[Telegram 전송 실패] 메시지 발송 실패 (토큰/채팅ID 및 네트워크 상태를 확인하세요, chat_id={self.chat_id})")
        return ok_all

    def send_signal_alert(self, signal: dict[str, Any]) -> bool:
        return self.send_text(format_signal_message(signal))

    def send_daily_analysis(
        self,
        error_result: dict[str, Any],
        top20_result: dict[str, Any],
        learning_result: dict[str, Any] | None = None,
    ) -> bool:
        return self.send_text(format_top20_analysis_message(error_result, top20_result, learning_result))

    def send_weekly_report(self, report: dict[str, Any]) -> bool:
        return self.send_text(format_weekly_report_message(report))

    def send_closing_briefing(self, briefing: dict[str, Any]) -> bool:
        return self.send_text(format_closing_briefing_message(briefing))
