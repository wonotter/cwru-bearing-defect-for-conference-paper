"""
CWRU .mat 파일을 로드하고 DE(Drive End) 가속도계 시계열을 추출한다.

정상 데이터(48 kHz)는 12 kHz로 다운샘플링하여 결함 데이터와 동일한
샘플링 레이트로 통일한다.
"""

from __future__ import annotations

import numpy as np
import scipy.io
from scipy.signal import resample_poly

from cwru_analysis.catalog import DatasetEntry
from cwru_analysis.config import ANALYSIS_FS, NORMAL_FS


# ─────────────────────────────────────────────────────────────────
# 반환 타입 (dict 기반, 모든 모듈이 동일한 형식 사용)
# ─────────────────────────────────────────────────────────────────
# {
#   "key":        str       — 데이터셋 고유 식별자
#   "signal":     ndarray   — DE 가속도계 시계열 (1D, float64)
#   "fs":         int       — 분석용 샘플링 레이트 (Hz), 항상 12000
#   "rpm":        float     — 분석에 사용할 RPM (.mat 실측 우선)
#   "rpm_source": str       — "file" 또는 "nominal"
#   "n_samples":  int       — 시계열 샘플 수
#   "duration_s": float     — 시계열 길이 (초)
#   "info":       DatasetEntry — 카탈로그 메타 정보
# }


def load_dataset(entry: DatasetEntry) -> dict:
    """
    카탈로그 엔트리 하나를 로드하여 분석용 딕셔너리를 반환한다.

    정상 데이터(48 kHz)는 안티앨리어싱 필터와 함께 12 kHz로 다운샘플링한다.
    결함 데이터(12 kHz)는 그대로 사용한다.

    Parameters
    ----------
    entry : DatasetEntry
        catalog.build_catalog()에서 얻은 엔트리

    Returns
    -------
    dict
        분석용 데이터 딕셔너리

    Raises
    ------
    FileNotFoundError
        .mat 파일이 존재하지 않을 때
    KeyError
        .mat 파일에서 DE 시계열 키를 찾을 수 없을 때
    """
    mat_path = entry.file_path
    if not mat_path.is_file():
        raise FileNotFoundError(f".mat 파일을 찾을 수 없습니다: {mat_path}")

    mat = scipy.io.loadmat(str(mat_path))

    # DE 시계열 추출
    de_key = _find_de_key(mat, entry.file_id)
    signal = mat[de_key].flatten().astype(np.float64)

    # RPM 추출 (.mat 실측값 우선, 없으면 공칭값 폴백)
    rpm_measured = _find_rpm(mat, entry.file_id)
    if rpm_measured is not None:
        rpm = float(rpm_measured)
        rpm_source = "file"
    else:
        rpm = float(entry.nominal_rpm)
        rpm_source = "nominal"

    # 48 kHz → 12 kHz 다운샘플링 (정상 데이터)
    original_fs = entry.fs
    if original_fs == NORMAL_FS:
        signal = _resample_to_analysis_fs(signal, original_fs)
    elif original_fs != ANALYSIS_FS:
        raise ValueError(
            f"지원하지 않는 샘플링 레이트: {original_fs} Hz "
            f"(파일: {entry.key})"
        )

    n_samples = len(signal)
    duration_s = n_samples / ANALYSIS_FS

    return {
        "key": entry.key,
        "signal": signal,
        "fs": ANALYSIS_FS,
        "rpm": rpm,
        "rpm_source": rpm_source,
        "n_samples": n_samples,
        "duration_s": duration_s,
        "info": entry,
    }


def load_datasets(entries: dict[str, DatasetEntry]) -> dict[str, dict]:
    """
    여러 데이터셋을 일괄 로드한다.

    Parameters
    ----------
    entries : dict[str, DatasetEntry]
        {key: DatasetEntry} 딕셔너리

    Returns
    -------
    dict[str, dict]
        {key: load_dataset(entry)} 딕셔너리
    """
    return {key: load_dataset(entry) for key, entry in entries.items()}


# ─────────────────────────────────────────────────────────────────
# 다운샘플링
# ─────────────────────────────────────────────────────────────────
def _resample_to_analysis_fs(signal: np.ndarray, original_fs: int) -> np.ndarray:
    """
    48 kHz 신호를 12 kHz로 다운샘플링한다.

    scipy.signal.resample_poly는 내부적으로 안티앨리어싱 FIR 필터를
    적용한 뒤 정수 비율(up/down)로 리샘플한다.
    48000 / 12000 = 4이므로 up=1, down=4.

    Parameters
    ----------
    signal : np.ndarray
        원본 신호 (1D)
    original_fs : int
        원본 샘플링 레이트

    Returns
    -------
    np.ndarray
        12 kHz로 다운샘플링된 신호
    """
    if original_fs == ANALYSIS_FS:
        return signal
    if original_fs == NORMAL_FS:
        return resample_poly(signal, up=1, down=4)
    raise ValueError(f"지원하지 않는 원본 샘플링 레이트: {original_fs} Hz")


# ─────────────────────────────────────────────────────────────────
# .mat 파일 키 탐색
# ─────────────────────────────────────────────────────────────────
def _find_de_key(mat: dict, file_id: int) -> str:
    """
    .mat 파일에서 Drive End 시계열 변수 키를 탐색한다.

    CWRU 형식: "X{file_id}_DE_time" 또는 "X{file_id:03d}_DE_time"

    Parameters
    ----------
    mat : dict
        scipy.io.loadmat()으로 로드된 딕셔너리
    file_id : int
        CWRU Recording ID

    Returns
    -------
    str
        매칭된 DE 키

    Raises
    ------
    KeyError
        DE 키를 찾을 수 없을 때
    """
    candidates = [
        f"X{file_id:03d}_DE_time",
        f"X{file_id}_DE_time",
    ]
    for key in candidates:
        if key in mat:
            return key

    # 유연 탐색: 'DE_time' 포함된 키
    for key in mat:
        if "DE_time" in key:
            return key

    available = [k for k in mat if not k.startswith("__")]
    raise KeyError(
        f"파일 ID {file_id}에서 DE 시계열을 찾을 수 없습니다. "
        f"사용 가능한 키: {available}"
    )


def _find_rpm(mat: dict, file_id: int) -> float | None:
    """
    .mat 파일에 기록된 실측 RPM을 읽는다.

    CWRU 파일 일부에는 RPM 변수가 없으므로 없으면 None을 반환한다.

    Parameters
    ----------
    mat : dict
        scipy.io.loadmat()으로 로드된 딕셔너리
    file_id : int
        CWRU Recording ID

    Returns
    -------
    float | None
        실측 RPM, 또는 변수가 없으면 None
    """
    candidates = [f"X{file_id:03d}RPM", f"X{file_id}RPM"]
    # RPM으로 끝나는 키를 추가 탐색
    candidates += [k for k in mat if k.endswith("RPM") and not k.startswith("__")]

    for key in candidates:
        if key in mat:
            values = np.asarray(mat[key]).flatten()
            if values.size > 0:
                return float(values[0])

    return None


# ─────────────────────────────────────────────────────────────────
# 요약 출력
# ─────────────────────────────────────────────────────────────────
def print_dataset_summary(data: dict) -> None:
    """로드된 데이터셋의 요약을 콘솔에 출력한다."""
    entry: DatasetEntry = data["info"]
    original_fs = entry.fs
    resampled = original_fs != ANALYSIS_FS

    print(f"  {data['key']:<16s}  "
          f"ID={entry.file_id:<4d}  "
          f"type={entry.fault_type:<13s}  "
          f"size={entry.fault_diameter or '-':>5s}  "
          f"HP={entry.hp}  "
          f"RPM={data['rpm']:>6.1f}({data['rpm_source']})  "
          f"fs={data['fs']}Hz"
          f"{'<-' + str(original_fs) if resampled else '':>8s}  "
          f"samples={data['n_samples']:>7,d}  "
          f"dur={data['duration_s']:.2f}s")
