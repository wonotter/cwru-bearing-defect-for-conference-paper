"""
성능 평가 모듈 (Phase 5).

분류 결과에 대한 Accuracy, Precision, Recall, F1, 혼동행렬,
ROC/AUC 등을 계산하고, 그래프와 JSON/CSV 파일로 저장한다.

평가 단위
──────────
  - **조건별 개별 평가**: 12개 조건 (결함크기 3 × HP 4) 각각
  - **전체 통합 평가**: 모든 구간을 합산한 실험 A / B 전체

저장 구조
──────────
  results/classification/
  ├── with_ball/           (실험 A)
  │   ├── per_condition/
  │   │   ├── 0.007_0HP/  → metrics.json, segment_predictions.csv,
  │   │   │                  confusion_matrix.png, f1_scores.png
  │   │   └── ...
  │   ├── overall_metrics.json
  │   ├── overall_predictions.csv
  │   ├── overall_confusion_matrix.png
  │   ├── overall_f1_scores.png
  │   ├── overall_roc_curves.png
  │   └── accuracy_heatmap.png
  └── without_ball/        (실험 B, 동일 구조)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

from cwru_analysis.classifier import SegmentResult
from cwru_analysis.config import (
    FAULT_BALL,
    FAULT_INNER,
    FAULT_NORMAL,
    FAULT_OUTER,
    FIGURE_DPI,
    NORMAL_THRESHOLD,
    RESULTS_DIR,
)


# ─────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────
CLASSIFICATION_DIR = RESULTS_DIR / "classification"
CLASSES_WITH_BALL = [FAULT_OUTER, FAULT_INNER, FAULT_BALL, FAULT_NORMAL]
CLASSES_WITHOUT_BALL = [FAULT_OUTER, FAULT_INNER, FAULT_NORMAL]

_CLASS_COLORS = {
    FAULT_OUTER:  "#E74C3C",
    FAULT_INNER:  "#3498DB",
    FAULT_BALL:   "#2ECC71",
    FAULT_NORMAL: "#95A5A6",
}

_FAULT_TO_SCORE_KEY = {
    FAULT_OUTER: "BPFO",
    FAULT_INNER: "BPFI",
    FAULT_BALL:  "BSF",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# 메트릭 계산
# ─────────────────────────────────────────────────────────────────
def compute_metrics(
    results: list[SegmentResult],
    class_names: list[str],
) -> dict:
    """
    분류 결과에서 주요 성능 지표를 계산한다.

    Parameters
    ----------
    results : list[SegmentResult]
        분류 결과 리스트
    class_names : list[str]
        평가에 사용할 클래스 이름 목록 (순서 유지)

    Returns
    -------
    dict
        accuracy, n_samples, precision, recall, f1, macro_f1,
        support, confusion_matrix, class_names
    """
    y_true = [r.true_label for r in results]
    y_pred = [r.predicted_label for r in results]

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    return {
        "accuracy": acc,
        "n_samples": len(results),
        "precision": {c: float(p) for c, p in zip(class_names, prec)},
        "recall": {c: float(r) for c, r in zip(class_names, rec)},
        "f1": {c: float(f) for c, f in zip(class_names, f1)},
        "macro_f1": float(np.mean(f1)),
        "support": {c: int(s) for c, s in zip(class_names, sup)},
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


# ─────────────────────────────────────────────────────────────────
# 그래프: 혼동행렬
# ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    metrics: dict,
    title: str,
    save_path: Path,
) -> None:
    """혼동행렬 히트맵을 생성·저장한다."""
    cm = np.array(metrics["confusion_matrix"])
    class_names = metrics["class_names"]
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted", ylabel="True", title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight="bold",
            )

    fig.tight_layout()
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 그래프: F1 Score 바 차트
# ─────────────────────────────────────────────────────────────────
def plot_f1_scores(
    metrics: dict,
    title: str,
    save_path: Path,
) -> None:
    """클래스별 F1 Score 수평 바 차트를 생성·저장한다."""
    class_names = metrics["class_names"]
    f1_values = [metrics["f1"][c] for c in class_names]
    colors = [_CLASS_COLORS.get(c, "#888888") for c in class_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        class_names, f1_values,
        color=colors, edgecolor="black", linewidth=0.5,
    )

    for bar, val in zip(bars, f1_values):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=10,
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 Score")
    ax.set_title(
        f"{title}\n"
        f"Macro F1 = {metrics['macro_f1']:.4f}  |  "
        f"Accuracy = {metrics['accuracy']:.4f}"
    )
    ax.axvline(
        metrics["macro_f1"], color="gray", linestyle="--", alpha=0.6,
        label=f"Macro F1 = {metrics['macro_f1']:.3f}",
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 그래프: ROC 곡선 (One-vs-Rest)
# ─────────────────────────────────────────────────────────────────
def plot_roc_curves(
    results: list[SegmentResult],
    class_names: list[str],
    title: str,
    save_path: Path,
    normal_threshold: float = NORMAL_THRESHOLD,
) -> None:
    """
    One-vs-Rest ROC 곡선을 그린다.

    연속 점수 구성:
      - Outer Race → BPFO 점수
      - Inner Race → BPFI 점수
      - Ball       → BSF 점수
      - Normal     → threshold − max(scores)
    """
    y_true = [r.true_label for r in results]

    fig, ax = plt.subplots(figsize=(7, 6))

    for cls in class_names:
        binary_true = np.array([1 if y == cls else 0 for y in y_true])
        if len(np.unique(binary_true)) < 2:
            continue

        if cls in _FAULT_TO_SCORE_KEY:
            score_key = _FAULT_TO_SCORE_KEY[cls]
            class_scores = np.array([
                r.scores.get(score_key, 0.0) for r in results
            ])
        else:
            class_scores = np.array([
                normal_threshold - max(r.scores.values()) if r.scores else 0.0
                for r in results
            ])

        fpr, tpr, _ = roc_curve(binary_true, class_scores)
        roc_auc = auc(fpr, tpr)
        color = _CLASS_COLORS.get(cls, "#888888")

        ax.plot(
            fpr, tpr, color=color, linewidth=1.5,
            label=f"{cls} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 그래프: 조건별 Accuracy 히트맵
# ─────────────────────────────────────────────────────────────────
def plot_accuracy_heatmap(
    condition_metrics: dict[tuple[str, int], dict],
    title: str,
    save_path: Path,
) -> None:
    """
    결함 크기 × HP 조건별 정확도 히트맵을 그린다.

    행(y축) = 결함 크기, 열(x축) = HP
    """
    diameters = sorted({d for d, _ in condition_metrics})
    hps = sorted({h for _, h in condition_metrics})

    grid = np.zeros((len(diameters), len(hps)))
    for i, d in enumerate(diameters):
        for j, h in enumerate(hps):
            if (d, h) in condition_metrics:
                grid[i, j] = condition_metrics[(d, h)]["accuracy"]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Accuracy")

    ax.set_xticks(range(len(hps)))
    ax.set_xticklabels([f"{h}HP" for h in hps])
    ax.set_yticks(range(len(diameters)))
    ax.set_yticklabels([f'{d}"' for d in diameters])
    ax.set_xlabel("Motor Load")
    ax.set_ylabel("Fault Diameter")

    for i in range(len(diameters)):
        for j in range(len(hps)):
            ax.text(
                j, i, f"{grid[i, j]:.2f}",
                ha="center", va="center", fontsize=11,
                color="white" if grid[i, j] < 0.5 else "black",
            )

    ax.set_title(title)
    fig.tight_layout()

    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 파일 저장: JSON, CSV
# ─────────────────────────────────────────────────────────────────
def save_metrics_json(metrics: dict, save_path: Path) -> None:
    """메트릭을 JSON 파일로 저장한다."""
    _ensure_dir(save_path.parent)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_predictions_csv(results: list[SegmentResult], save_path: Path) -> None:
    """구간별 예측 결과를 CSV 파일로 저장한다."""
    _ensure_dir(save_path.parent)
    header = (
        "key,segment_idx,true_label,predicted_label,"
        "BPFO_score,BPFI_score,BSF_score,noise,fault_diameter,hp"
    )
    lines = [header]
    for r in results:
        bpfo = f"{r.scores['BPFO']:.4f}" if "BPFO" in r.scores else ""
        bpfi = f"{r.scores['BPFI']:.4f}" if "BPFI" in r.scores else ""
        bsf = f"{r.scores['BSF']:.4f}" if "BSF" in r.scores else ""
        fd = r.fault_diameter if r.fault_diameter else ""
        lines.append(
            f"{r.key},{r.segment_idx},{r.true_label},{r.predicted_label},"
            f"{bpfo},{bpfi},{bsf},{r.noise:.6f},{fd},{r.hp}"
        )
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────
# 통합 평가 실행
# ─────────────────────────────────────────────────────────────────
def run_evaluation(
    results: list[SegmentResult],
    include_ball: bool,
    experiment_label: str,
) -> dict:
    """
    한 실험(A 또는 B)의 전체 평가를 수행하고 결과를 저장한다.

    수행 내용:
      1. 조건별(결함크기 × HP) 메트릭 계산 + 그래프 + JSON/CSV 저장
      2. 전체 통합 메트릭 계산 + 그래프 + JSON/CSV 저장

    Parameters
    ----------
    results : list[SegmentResult]
        run_experiment()에서 반환된 전체 분류 결과
    include_ball : bool
        True → 실험 A (4종), False → 실험 B (3종)
    experiment_label : str
        "with_ball" 또는 "without_ball"

    Returns
    -------
    dict
        {
            "overall": dict (전체 메트릭),
            "per_condition": {(diameter, hp): dict (조건별 메트릭)},
        }
    """
    from cwru_analysis.catalog import get_all_conditions

    exp_dir = CLASSIFICATION_DIR / experiment_label
    class_names = CLASSES_WITH_BALL if include_ball else CLASSES_WITHOUT_BALL
    exp_display = "A (볼 포함)" if include_ball else "B (볼 제외)"
    exp_plot_label = "A (with Ball, 4-class)" if include_ball else "B (without Ball, 3-class)"

    print(f"\n  [평가] 실험 {exp_display}")

    # ── 1. 전체 통합 평가 ──
    overall = compute_metrics(results, class_names)
    print(f"    전체: Accuracy={overall['accuracy']:.4f}, "
          f"Macro F1={overall['macro_f1']:.4f} "
          f"(N={overall['n_samples']})")

    save_metrics_json(overall, exp_dir / "overall_metrics.json")
    save_predictions_csv(results, exp_dir / "overall_predictions.csv")

    plot_confusion_matrix(
        overall,
        f"Confusion Matrix \u2014 Experiment {exp_plot_label}",
        exp_dir / "overall_confusion_matrix.png",
    )
    plot_f1_scores(
        overall,
        f"F1 Scores \u2014 Experiment {exp_plot_label}",
        exp_dir / "overall_f1_scores.png",
    )
    plot_roc_curves(
        results, class_names,
        f"ROC Curves \u2014 Experiment {exp_plot_label}",
        exp_dir / "overall_roc_curves.png",
    )

    # ── 2. 조건별 개별 평가 ──
    per_condition: dict[tuple[str, int], dict] = {}

    for diameter, hp in get_all_conditions():
        cond_results = [
            r for r in results
            if (r.fault_diameter == diameter and r.hp == hp)
            or (r.fault_diameter is None and r.hp == hp)
        ]
        if not cond_results:
            continue

        cond_metrics = compute_metrics(cond_results, class_names)
        per_condition[(diameter, hp)] = cond_metrics

        cond_dir = exp_dir / "per_condition" / f"{diameter}_{hp}HP"
        cond_label = f'{diameter}" / {hp}HP'

        save_metrics_json(cond_metrics, cond_dir / "metrics.json")
        save_predictions_csv(cond_results, cond_dir / "segment_predictions.csv")

        plot_confusion_matrix(
            cond_metrics,
            f"Confusion Matrix \u2014 {cond_label}",
            cond_dir / "confusion_matrix.png",
        )
        plot_f1_scores(
            cond_metrics,
            f"F1 Scores \u2014 {cond_label}",
            cond_dir / "f1_scores.png",
        )

        print(f"    {cond_label}: Accuracy={cond_metrics['accuracy']:.4f}, "
              f"Macro F1={cond_metrics['macro_f1']:.4f} "
              f"(N={cond_metrics['n_samples']})")

    # ── 3. 조건별 Accuracy 히트맵 ──
    if per_condition:
        plot_accuracy_heatmap(
            per_condition,
            f"Accuracy Heatmap \u2014 Experiment {exp_plot_label}",
            exp_dir / "accuracy_heatmap.png",
        )

    print(f"    → 결과 저장 완료: {exp_dir}")
    return {"overall": overall, "per_condition": per_condition}
