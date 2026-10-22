"""
포락선(Envelope) 스펙트럼 계산.

파이프라인
──────────
1. Butterworth 밴드패스 필터 (Kurtogram이 선택한 대역)
2. 힐베르트 변환 -> 해석 신호
3. 일반 엔벨로프 |analytic|  (제곱하지 않음)
4. DC 제거 (평균 차감)
5. FFT -> 포락선 스펙트럼 (진폭, g 단위)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from cwru_analysis.config import ENVELOPE_FILTER_ORDER


# ─────────────────────────────────────────────────────────────────
# 밴드패스 필터
# ─────────────────────────────────────────────────────────────────
def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = ENVELOPE_FILTER_ORDER,
) -> np.ndarray:
    """
    Butterworth 밴드패스 필터를 영위상(sosfiltfilt)으로 적용한다.

    Parameters
    ----------
    signal : np.ndarray
        입력 신호 (1D)
    fs : float
        샘플링 레이트 (Hz)
    low, high : float
        통과 대역 하한/상한 (Hz)
    order : int
        필터 차수

    Returns
    -------
    np.ndarray
        필터링된 신호
    """
    nyquist = fs / 2.0
    low_n = max(low / nyquist, 1e-4)
    high_n = min(high / nyquist, 0.9999)
    if low_n >= high_n:
        low_n = high_n * 0.5

    sos = butter(order, [low_n, high_n], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


# ─────────────────────────────────────────────────────────────────
# 포락선 추출
# ─────────────────────────────────────────────────────────────────
def extract_envelope(signal: np.ndarray) -> np.ndarray:
    """힐베르트 변환으로 일반 엔벨로프 |analytic| 을 추출한다."""
    return np.abs(hilbert(signal))


# ─────────────────────────────────────────────────────────────────
# FFT (단측 진폭 스펙트럼)
# ─────────────────────────────────────────────────────────────────
def compute_fft(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    실수 신호의 단측 FFT 진폭 스펙트럼을 계산한다.

    Returns
    -------
    freqs : np.ndarray
        주파수 축 (Hz)
    magnitude : np.ndarray
        정규화 진폭 (2/N)
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(signal)
    magnitude = (2.0 / n) * np.abs(fft_vals)
    magnitude[0] /= 2.0  # DC는 1/N
    return freqs, magnitude


# ─────────────────────────────────────────────────────────────────
# 포락선 스펙트럼 (핵심 함수)
# ─────────────────────────────────────────────────────────────────
def compute_envelope_spectrum(
    signal: np.ndarray,
    fs: float,
    band: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    포락선 스펙트럼을 계산한다.

    band가 주어지면 Kurtogram이 선택한 대역으로 밴드패스 필터링 후
    포락선을 추출하고, band=None이면 밴드패스 없이 원 신호에서
    직접 포락선을 추출한다 (정상 데이터에 사용).

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 신호 (1D)
    fs : float
        샘플링 레이트 (Hz)
    band : tuple[float, float] | None
        (low_hz, high_hz) -- Kurtogram에서 선택한 최적 대역.
        None이면 밴드패스 필터를 적용하지 않는다.

    Returns
    -------
    freqs : np.ndarray
        주파수 축 (Hz)
    magnitude : np.ndarray
        포락선 스펙트럼 진폭 (g)
    envelope : np.ndarray
        시간 영역 포락선 (그래프용)
    """
    if band is not None:
        low, high = _clip_band(band[0], band[1], fs)
        filtered = bandpass_filter(signal, fs, low, high)
    else:
        filtered = signal
    envelope = extract_envelope(filtered)
    dc_removed = envelope - np.mean(envelope)
    freqs, magnitude = compute_fft(dc_removed, fs)
    return freqs, magnitude, envelope


# ─────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────
def _clip_band(low: float, high: float, fs: float) -> tuple[float, float]:
    """밴드패스 경계를 (0, 나이퀴스트) 안으로 클리핑한다."""
    nyquist = fs / 2.0
    high = min(high, nyquist * 0.99)
    low = max(low, 1.0)
    if low >= high:
        low = high * 0.5
    return low, high
