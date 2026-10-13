"""
규칙 기반 분류기 (Phase 4).

포락선 스펙트럼에서 결함 주파수 고조파 피크를 추출하고,
배경 노이즈 대비 비율(SNR)을 점수로 환산하여 결함 유형을 판정한다.
**학습 과정이 없으며**, 모든 파라미터는 사전에 고정된다.

분류 흐름
──────────
  1. 파일 전체 신호 → Fast Kurtogram → 최적 밴드패스 대역 (파일당 1회)
  2. 신호를 1초 구간으로 분할
  3. 구간별 밴드패스 → 힐베르트 → 포락선 → FFT
  4. BPFO / BPFI / BSF 고조파 위치에서 피크 진폭 → 배경 노이즈로 나누기 → 평균
  5. 판정:
     - 실험 A (include_ball=True):  모든 점수 < 임계값 → Normal, else argmax
     - 실험 B (include_ball=False): BPFO, BPFI만 비교
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cwru_analysis.config import (
    FAULT_BALL,
    FAULT_INNER,
    FAULT_NORMAL,
    FAULT_OUTER,
    N_HARMONICS_BPFI,
    N_HARMONICS_BPFO,
    N_HARMONICS_BSF,
    NORMAL_THRESHOLD,
    SEGMENT_SECONDS,
)
from cwru_analysis.envelope import compute_envelope_spectrum
from cwru_analysis.frequencies import calculate_defect_frequencies


# ─────────────────────────────────────────────────────────────────
# 분류기 설정 (학습 없이 고정)
# ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ClassifierConfig:
    """
    분류기 설정.

    모든 데이터에 공통 적용되며, 학습 과정 없이 사전에 결정된
    파라미터만 사용한다.

    Attributes
    ----------
    n_harmonics_bpfo : int
        BPFO 고조파 탐색 개수 (기본 3)
    n_harmonics_bpfi : int
        BPFI 고조파 탐색 개수 (기본 3)
    n_harmonics_bsf : int
        BSF 고조파 탐색 개수 (기본 6, 짝수 포함)
    relative_tolerance : float
        피크 검색 허용 범위 — 해당 주파수의 ±비율 (0.02 = ±2%)
    tolerance_bins : float
        최소 검색 반폭 — 주파수 분해능의 배수 (분해능 × 2)
    noise_low_hz : float
        배경 노이즈 계산 주파수 하한 (Hz)
    noise_high_hz : float
        배경 노이즈 계산 주파수 상한 (Hz)
    segment_seconds : float
        신호 분할 구간 길이 (초)
    normal_threshold : float
        정상 판정 임계값 — 모든 점수가 이 값 미만이면 Normal
    include_ball : bool
        True → 실험 A (4종 분류), False → 실험 B (3종 분류, BSF 무시)
    """

    n_harmonics_bpfo: int = N_HARMONICS_BPFO
    n_harmonics_bpfi: int = N_HARMONICS_BPFI
    n_harmonics_bsf: int = N_HARMONICS_BSF
    relative_tolerance: float = 0.02
    tolerance_bins: float = 2.0
    noise_low_hz: float = 20.0
    noise_high_hz: float = 500.0
    segment_seconds: float = SEGMENT_SECONDS
    normal_threshold: float = NORMAL_THRESHOLD
    include_ball: bool = True


# ─────────────────────────────────────────────────────────────────
# 분류 결과 자료구조
# ─────────────────────────────────────────────────────────────────
@dataclass
class SegmentResult:
    """1초 구간 하나의 분류 결과."""

    key: str                          # 데이터셋 키 (예: "OR007@6_0")
    segment_idx: int                  # 구간 인덱스 (0-based)
    true_label: str                   # 실제 결함 유형
    predicted_label: str              # 분류기 판정 결과
    scores: dict[str, float]          # {"BPFO": ..., "BPFI": ..., "BSF": ...}
    noise: float                      # 배경 노이즈 수준
    fault_diameter: str | None        # 결함 크기 (정상이면 None)
    hp: int                           # 모터 부하 (0~3)


# 결함 주파수 이름 → 결함 유형 문자열
_FREQ_TO_FAULT = {
    "BPFO": FAULT_OUTER,
    "BPFI": FAULT_INNER,
    "BSF":  FAULT_BALL,
}


# ─────────────────────────────────────────────────────────────────
# 1초 구간 분류
# ─────────────────────────────────────────────────────────────────
def classify_segment(
    segment: np.ndarray,
    fs: float,
    rpm: float,
    band: tuple[float, float],
    config: ClassifierConfig,
) -> dict:
    """
    1초 구간 하나를 분류한다.

    Parameters
    ----------
    segment : np.ndarray
        1초 길이의 시간 영역 신호
    fs : float
        샘플링 레이트 (Hz)
    rpm : float
        모터 RPM
    band : tuple[float, float]
        Kurtogram 밴드패스 대역 (low_hz, high_hz)
    config : ClassifierConfig

    Returns
    -------
    dict
        {"predicted": str, "scores": dict, "noise": float}
    """
    freqs, magnitude, _ = compute_envelope_spectrum(segment, fs, band)
    freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    defect = calculate_defect_frequencies(
        rpm, config.n_harmonics_bpfo,
        config.n_harmonics_bpfi, config.n_harmonics_bsf,
    )

    noise = _calculate_noise(freqs, magnitude, defect, freq_res, config)

    score_names = ["BPFO", "BPFI"]
    if config.include_ball:
        score_names.append("BSF")

    scores: dict[str, float] = {}
    for fname in score_names:
        scores[fname] = _calculate_score(
            freqs, magnitude, defect[fname], noise, freq_res, config,
        )

    max_score = max(scores.values()) if scores else 0.0
    if max_score < config.normal_threshold:
        predicted = FAULT_NORMAL
    else:
        best_fname = max(scores, key=scores.get)
        predicted = _FREQ_TO_FAULT[best_fname]

    return {"predicted": predicted, "scores": scores, "noise": noise}


# ─────────────────────────────────────────────────────────────────
# 파일 단위 분류
# ─────────────────────────────────────────────────────────────────
def classify_file(
    data: dict,
    band: tuple[float, float],
    config: ClassifierConfig,
) -> list[SegmentResult]:
    """
    파일 전체를 1초 구간으로 분할하고 각각 분류한다.

    Kurtogram 대역(band)은 파일 전체 신호에서 미리 계산되어 전달된다.
    해당 파일의 모든 1초 구간에 동일한 대역을 적용한다.

    Parameters
    ----------
    data : dict
        load_dataset()으로 로드된 데이터
    band : tuple[float, float]
        해당 파일의 Kurtogram 최적 대역
    config : ClassifierConfig

    Returns
    -------
    list[SegmentResult]
    """
    signal = data["signal"]
    fs = data["fs"]
    rpm = data["rpm"]
    info = data["info"]

    segment_length = int(config.segment_seconds * fs)
    n_segments = len(signal) // segment_length

    if n_segments == 0:
        print(f"    [경고] {data['key']}: 신호 길이({len(signal)})가 "
              f"1초 구간({segment_length}) 미만. 건너뜀.")
        return []

    results: list[SegmentResult] = []
    for i in range(n_segments):
        start = i * segment_length
        segment = signal[start:start + segment_length]

        out = classify_segment(segment, fs, rpm, band, config)
        results.append(SegmentResult(
            key=data["key"],
            segment_idx=i,
            true_label=info.fault_type,
            predicted_label=out["predicted"],
            scores=dict(out["scores"]),
            noise=out["noise"],
            fault_diameter=info.fault_diameter,
            hp=info.hp,
        ))

    return results


# ─────────────────────────────────────────────────────────────────
# 실험 실행
# ─────────────────────────────────────────────────────────────────
def run_experiment(
    all_data: dict[str, dict],
    bands: dict[str, tuple[float, float]],
    include_ball: bool = True,
) -> list[SegmentResult]:
    """
    실험 A(볼 포함) 또는 B(볼 제외)를 전체 데이터에 대해 실행한다.

    Parameters
    ----------
    all_data : dict[str, dict]
        {key: load_dataset()} 딕셔너리
    bands : dict[str, tuple[float, float]]
        {key: (low_hz, high_hz)} — 파일별 Kurtogram 대역
    include_ball : bool
        True = 실험 A (4종 분류), False = 실험 B (3종 분류)

    Returns
    -------
    list[SegmentResult]
        전체 구간의 분류 결과 리스트
    """
    config = ClassifierConfig(include_ball=include_ball)
    exp_name = "A (볼 포함, 4종)" if include_ball else "B (볼 제외, 3종)"

    all_results: list[SegmentResult] = []
    skipped = 0

    for key, data in all_data.items():
        if not include_ball and data["info"].fault_type == FAULT_BALL:
            skipped += 1
            continue

        band = bands[key]
        file_results = classify_file(data, band, config)
        all_results.extend(file_results)

    n_correct = sum(1 for r in all_results if r.true_label == r.predicted_label)
    n_total = len(all_results)
    acc = n_correct / n_total if n_total > 0 else 0.0

    print(f"    실험 {exp_name}: "
          f"{n_total}개 구간 분류 완료 "
          f"(정확도 {acc:.4f}, {n_correct}/{n_total})")
    if skipped > 0:
        print(f"      → Ball 파일 {skipped}개 건너뜀")

    return all_results


# ─────────────────────────────────────────────────────────────────
# 내부 함수: 점수 계산
# ─────────────────────────────────────────────────────────────────
def _calculate_score(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    harmonics: list[float],
    noise: float,
    freq_resolution: float,
    config: ClassifierConfig,
) -> float:
    """
    특정 결함 유형의 고조파 평균 점수(SNR)를 계산한다.

    각 고조파 위치에서 허용 범위(tolerance) 내 최대 진폭을 찾고,
    배경 노이즈로 나눈 비율의 평균을 반환한다.

    Tolerance 계산
    ────────────────
      half_width = max(
          freq × relative_tolerance,     # 주파수 비례 (±2%)
          tolerance_bins × freq_res       # 최소 빈 수 보장 (±2빈)
      )
    """
    safe_noise = max(noise, 1e-10)
    ratios: list[float] = []

    for freq in harmonics:
        tol = max(
            freq * config.relative_tolerance,
            config.tolerance_bins * freq_resolution,
        )
        mask = (freqs >= freq - tol) & (freqs <= freq + tol)
        peak = float(np.max(magnitude[mask])) if np.any(mask) else 0.0
        ratios.append(peak / safe_noise)

    return float(np.mean(ratios)) if ratios else 0.0


def _calculate_noise(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    defect: dict,
    freq_resolution: float,
    config: ClassifierConfig,
) -> float:
    """
    배경 노이즈를 계산한다.

    20~500 Hz 범위에서 모든 결함 주파수 검색 영역을 **제외한** 뒤
    남은 주파수 빈의 진폭 **중앙값**을 반환한다.
    """
    mask = (freqs >= config.noise_low_hz) & (freqs <= config.noise_high_hz)

    for fname in ("BPFO", "BPFI", "BSF"):
        if fname not in defect:
            continue
        for freq in defect[fname]:
            tol = max(
                freq * config.relative_tolerance,
                config.tolerance_bins * freq_resolution,
            )
            exclude = (freqs >= freq - tol) & (freqs <= freq + tol)
            mask = mask & ~exclude

    if np.any(mask):
        return float(np.median(magnitude[mask]))

    fallback = (freqs >= config.noise_low_hz) & (freqs <= config.noise_high_hz)
    if np.any(fallback):
        return float(np.median(magnitude[fallback]))

    return 1e-10
