"""
RPM에서 각 결함 유형의 주파수와 고조파를 계산한다.

BSF(Ball Spin Frequency)는 볼 자전 주파수(2.357 x shaft)를 기본으로 두고
1~6차 고조파를 모두 산출한다.  짝수 고조파(2xBSF, 4xBSF)는 볼이 1회전할 때
내륜·외륜에 각 1회씩 충격을 주기 때문에 실제 스펙트럼에서 홀수보다
강하게 나타나는 경우가 많다.
"""

from __future__ import annotations

from cwru_analysis.config import (
    DEFECT_FREQ_MULTIPLIERS,
    N_HARMONICS_BPFI,
    N_HARMONICS_BPFO,
    N_HARMONICS_BSF,
)


def calculate_defect_frequencies(
    rpm: float,
    n_harmonics_bpfo: int = N_HARMONICS_BPFO,
    n_harmonics_bpfi: int = N_HARMONICS_BPFI,
    n_harmonics_bsf: int = N_HARMONICS_BSF,
) -> dict:
    """
    주어진 RPM에서 축 회전 주파수와 결함 주파수 고조파를 계산한다.

    Parameters
    ----------
    rpm : float
        모터 회전 속도 (RPM)
    n_harmonics_bpfo : int
        BPFO 고조파 개수 (기본 3)
    n_harmonics_bpfi : int
        BPFI 고조파 개수 (기본 3)
    n_harmonics_bsf : int
        BSF 고조파 개수 (기본 6 — 짝수 고조파 포함)

    Returns
    -------
    dict
        {
            "shaft_freq": float,
            "BPFO": [1차, 2차, 3차],
            "BPFI": [1차, 2차, 3차],
            "BSF":  [1차, 2차, ..., 6차],
            "FTF":  [1차],
        }
    """
    shaft_freq = rpm / 60.0

    harmonics_map = {
        "BPFO": n_harmonics_bpfo,
        "BPFI": n_harmonics_bpfi,
        "BSF":  n_harmonics_bsf,
        "FTF":  1,
    }

    result: dict = {"shaft_freq": shaft_freq}
    for name, multiplier in DEFECT_FREQ_MULTIPLIERS.items():
        base_freq = multiplier * shaft_freq
        n = harmonics_map[name]
        result[name] = [base_freq * (i + 1) for i in range(n)]

    return result


def print_defect_frequencies(rpm: float) -> None:
    """결함 주파수 계산 결과를 콘솔에 출력한다."""
    freqs = calculate_defect_frequencies(rpm)
    shaft = freqs["shaft_freq"]

    print(f"\n{'=' * 60}")
    print(f"  결함 주파수 (RPM={rpm:.1f}, shaft={shaft:.3f} Hz)")
    print(f"{'=' * 60}")

    for name in ("BPFO", "BPFI", "BSF", "FTF"):
        harmonics = freqs[name]
        mult = DEFECT_FREQ_MULTIPLIERS[name]
        print(f"\n  {name} (x{mult})")
        for i, freq in enumerate(harmonics):
            tag = "even" if name == "BSF" and (i + 1) % 2 == 0 else ""
            order = f"{i + 1}x"
            print(f"    {order:>4s}: {freq:8.2f} Hz  {tag}")

    print(f"\n  1xRPM: {shaft:.3f} Hz")
    print(f"{'=' * 60}\n")
