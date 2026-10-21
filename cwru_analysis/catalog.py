"""
40개 CWRU 데이터셋을 폴더 구조에서 자동으로 등록하고 조건별로 필터링한다.

폴더 레이아웃
─────────────
dataset/
├── 12kHz Drive End/
│   ├── {fault_diameter}/          # "0.007", "0.014", "0.021"
│   │   └── {hp} HP/              # "0 HP" ~ "3 HP"
│   │       └── {key}/            # "OR007@6_0", "IR007_0", "B007_0" 등
│   │           └── {file_id}.mat
└── normal/
    └── normal_{hp}/              # "normal_0" ~ "normal_3"
        └── {file_id}.mat
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cwru_analysis.config import (
    ANALYSIS_FS,
    DATASET_DIR,
    FAULT_BALL,
    FAULT_DIAMETERS,
    FAULT_INNER,
    FAULT_NORMAL,
    FAULT_OUTER,
    HP_TO_RPM,
    NORMAL_FS,
)


# ─────────────────────────────────────────────────────────────────
# 데이터셋 엔트리
# ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DatasetEntry:
    """카탈로그에 등록된 하나의 데이터셋 정보."""

    key: str                   # 고유 식별자 (예: "OR007@6_0", "Normal_0")
    file_path: Path            # .mat 파일 절대 경로
    file_id: int               # CWRU Recording ID (파일명에서 추출)
    fault_type: str            # "Outer Race" | "Inner Race" | "Ball" | "Normal"
    fault_diameter: str | None # "0.007" | "0.014" | "0.021" | None(정상)
    hp: int                    # 모터 부하 (0~3)
    nominal_rpm: float         # HP_TO_RPM 공칭값
    fs: int                    # 원본 샘플링 레이트 (Hz)
    or_position: str | None    # 외륜이면 "6:00", 나머지 None


# ─────────────────────────────────────────────────────────────────
# 키 → 결함 유형 파싱 헬퍼
# ─────────────────────────────────────────────────────────────────
def _parse_fault_type(folder_name: str) -> str:
    """폴더 이름(예: "OR007@6_0")에서 결함 유형을 추출한다."""
    upper = folder_name.upper()
    if upper.startswith("OR"):
        return FAULT_OUTER
    if upper.startswith("IR"):
        return FAULT_INNER
    if upper.startswith("B"):
        return FAULT_BALL
    raise ValueError(f"결함 유형을 파싱할 수 없는 폴더 이름: {folder_name}")


def _parse_or_position(folder_name: str) -> str | None:
    """외륜 폴더에서 결함 위치를 추출한다. '@6' → '6:00'."""
    if "@6" in folder_name:
        return "6:00"
    if "@3" in folder_name:
        return "3:00"
    if "@12" in folder_name:
        return "12:00"
    return None


# ─────────────────────────────────────────────────────────────────
# 카탈로그 빌더
# ─────────────────────────────────────────────────────────────────
def _build_fault_entries(dataset_dir: Path) -> list[DatasetEntry]:
    """12kHz Drive End 결함 데이터를 폴더 구조에서 탐색하여 엔트리 목록을 만든다."""
    fault_root = dataset_dir / "12kHz Drive End"
    entries: list[DatasetEntry] = []

    if not fault_root.is_dir():
        return entries

    for size_dir in sorted(fault_root.iterdir()):
        if not size_dir.is_dir():
            continue
        fault_diameter = size_dir.name  # "0.007", "0.014", "0.021"

        # FAULT_DIAMETERS에 포함된 결함 크기만 등록
        if fault_diameter not in FAULT_DIAMETERS:
            continue

        for hp_dir in sorted(size_dir.iterdir()):
            if not hp_dir.is_dir():
                continue
            hp = int(hp_dir.name.replace(" HP", "").strip())

            for key_dir in sorted(hp_dir.iterdir()):
                if not key_dir.is_dir():
                    continue
                key = key_dir.name  # "OR007@6_0", "IR007_0", "B007_0"

                mat_files = list(key_dir.glob("*.mat"))
                if not mat_files:
                    continue
                mat_file = mat_files[0]
                file_id = int(mat_file.stem)

                entries.append(DatasetEntry(
                    key=key,
                    file_path=mat_file.resolve(),
                    file_id=file_id,
                    fault_type=_parse_fault_type(key),
                    fault_diameter=fault_diameter,
                    hp=hp,
                    nominal_rpm=HP_TO_RPM[hp],
                    fs=ANALYSIS_FS,
                    or_position=_parse_or_position(key),
                ))

    return entries


def _build_normal_entries(dataset_dir: Path) -> list[DatasetEntry]:
    """정상 베이스라인 데이터를 폴더 구조에서 탐색하여 엔트리 목록을 만든다."""
    normal_root = dataset_dir / "normal"
    entries: list[DatasetEntry] = []

    if not normal_root.is_dir():
        return entries

    for hp_dir in sorted(normal_root.iterdir()):
        if not hp_dir.is_dir():
            continue
        # "normal_0" → hp=0
        hp = int(hp_dir.name.split("_")[-1])

        mat_files = list(hp_dir.glob("*.mat"))
        if not mat_files:
            continue
        mat_file = mat_files[0]
        file_id = int(mat_file.stem)

        key = f"Normal_{hp}"
        entries.append(DatasetEntry(
            key=key,
            file_path=mat_file.resolve(),
            file_id=file_id,
            fault_type=FAULT_NORMAL,
            fault_diameter=None,
            hp=hp,
            nominal_rpm=HP_TO_RPM[hp],
            fs=NORMAL_FS,  # 정상은 48kHz 원본
            or_position=None,
        ))

    return entries


def build_catalog(dataset_dir: Path | None = None) -> dict[str, DatasetEntry]:
    """
    전체 데이터셋 카탈로그를 구축한다.

    Parameters
    ----------
    dataset_dir : Path | None
        데이터셋 루트 디렉토리. None이면 config.DATASET_DIR 사용.

    Returns
    -------
    dict[str, DatasetEntry]
        {key: DatasetEntry} 딕셔너리. key 알파벳 순 정렬.
    """
    if dataset_dir is None:
        dataset_dir = DATASET_DIR

    entries = _build_fault_entries(dataset_dir) + _build_normal_entries(dataset_dir)
    catalog = {entry.key: entry for entry in entries}

    return dict(sorted(catalog.items()))


# ─────────────────────────────────────────────────────────────────
# 카탈로그 필터링 유틸리티
# ─────────────────────────────────────────────────────────────────
def filter_catalog(
    catalog: dict[str, DatasetEntry],
    *,
    fault_types: list[str] | None = None,
    fault_diameter: str | None = None,
    hp: int | None = None,
    include_ball: bool = True,
) -> dict[str, DatasetEntry]:
    """
    조건에 맞는 엔트리만 필터링한다.

    Parameters
    ----------
    catalog : dict[str, DatasetEntry]
        전체 카탈로그
    fault_types : list[str] | None
        포함할 결함 유형 목록. None이면 모든 유형.
    fault_diameter : str | None
        결함 크기 필터. None이면 모든 크기. 정상은 None이므로 항상 포함.
    hp : int | None
        HP 필터. None이면 모든 HP.
    include_ball : bool
        False이면 Ball 유형 제외.

    Returns
    -------
    dict[str, DatasetEntry]
        필터링된 카탈로그
    """
    result = {}
    for key, entry in catalog.items():
        if not include_ball and entry.fault_type == FAULT_BALL:
            continue
        if fault_types is not None and entry.fault_type not in fault_types:
            continue
        if hp is not None and entry.hp != hp:
            continue
        if fault_diameter is not None:
            # 정상 데이터는 fault_diameter=None → 결함 크기 필터와 무관하게 포함
            if entry.fault_diameter is not None and entry.fault_diameter != fault_diameter:
                continue
        result[key] = entry
    return result


def get_experiment_datasets(
    catalog: dict[str, DatasetEntry],
    fault_diameter: str,
    hp: int,
    include_ball: bool = True,
) -> dict[str, DatasetEntry]:
    """
    하나의 실험 조건(결함 크기 + HP)에 해당하는 데이터셋을 반환한다.

    반환되는 데이터:
      - include_ball=True  : OR@6 + IR + Ball + Normal (4종)
      - include_ball=False : OR@6 + IR + Normal (3종)

    Parameters
    ----------
    catalog : dict[str, DatasetEntry]
    fault_diameter : str
        "0.007", "0.014", "0.021"
    hp : int
        0, 1, 2, 3
    include_ball : bool
        볼 결함 포함 여부

    Returns
    -------
    dict[str, DatasetEntry]
    """
    return filter_catalog(
        catalog,
        fault_diameter=fault_diameter,
        hp=hp,
        include_ball=include_ball,
    )


def get_all_conditions() -> list[tuple[str, int]]:
    """
    전체 실험 조건 목록을 반환한다.

    config.FAULT_DIAMETERS에 정의된 결함 크기만 포함한다.

    Returns
    -------
    list[tuple[str, int]]
        [(결함크기, HP), ...] — FAULT_DIAMETERS × HP 조합
    """
    hps = [0, 1, 2, 3]
    return [(d, h) for d in FAULT_DIAMETERS for h in hps]


def print_catalog_summary(catalog: dict[str, DatasetEntry]) -> None:
    """카탈로그 요약을 콘솔에 출력한다."""
    print(f"\n{'=' * 70}")
    print(f"  데이터셋 카탈로그: 총 {len(catalog)}개 파일")
    print(f"{'=' * 70}")

    # 결함 유형별 집계
    type_counts: dict[str, int] = {}
    for entry in catalog.values():
        type_counts[entry.fault_type] = type_counts.get(entry.fault_type, 0) + 1

    for fault_type, count in sorted(type_counts.items()):
        print(f"  {fault_type:>12s}: {count}개")

    print(f"\n  {'Key':<16s} {'Type':<13s} {'Size':>6s} {'HP':>3s} "
          f"{'RPM':>6s} {'fs':>6s} {'File ID':>8s}")
    print(f"  {'-' * 16} {'-' * 13} {'-' * 6} {'-' * 3} "
          f"{'-' * 6} {'-' * 6} {'-' * 8}")

    for entry in catalog.values():
        size_str = entry.fault_diameter or "  -"
        print(f"  {entry.key:<16s} {entry.fault_type:<13s} {size_str:>6s} "
              f"{entry.hp:>3d} {entry.nominal_rpm:>6.0f} "
              f"{entry.fs:>6d} {entry.file_id:>8d}")

    print(f"{'=' * 70}\n")
