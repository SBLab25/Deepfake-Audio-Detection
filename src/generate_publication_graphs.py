from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)


COLORS = [
    "#2563eb",
    "#f97316",
    "#10b981",
    "#d946ef",
    "#ef4444",
    "#14b8a6",
    "#f59e0b",
]


def set_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfbff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#0f172a",
            "axes.titlecolor": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#dbeafe",
            "grid.alpha": 0.85,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#cbd5e1",
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def parse_label(value: str) -> int:
    return 1 if str(value).strip().lower() == "fake" else 0


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_seed_metrics(artifact_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict] = []
    history_rows: list[dict] = []
    for seed_dir in sorted(artifact_dir.glob("seed_*")):
        metrics_path = seed_dir / "metrics_safari_wavlm_ensemble.json"
        if not metrics_path.exists():
            continue
        seed = int(seed_dir.name.split("_", 1)[1])
        metrics = load_json(metrics_path)
        seed_rows.append(
            {
                "seed": seed,
                "best_epoch": metrics.get("best_epoch"),
                "threshold": metrics.get("threshold"),
                "val_auc": metrics["val"]["metrics"]["roc_auc"],
                "val_accuracy": metrics["val"]["metrics"]["accuracy"],
                "val_f1": metrics["val"]["metrics"]["f1"],
                "test_auc": metrics["test"]["metrics"]["roc_auc"],
                "test_accuracy": metrics["test"]["metrics"]["accuracy"],
                "test_f1": metrics["test"]["metrics"]["f1"],
            }
        )
        for row in metrics.get("history", []):
            history_rows.append({"seed": seed, **row})
    return pd.DataFrame(seed_rows), pd.DataFrame(history_rows)


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    y_true = df["label"].map(parse_label).to_numpy(dtype=np.int64)
    y_score = df["fake_probability"].astype(float).to_numpy()
    return y_true, y_score


def infer_project_root(artifact_dir: Path) -> Path:
    if artifact_dir.name == "publication_challenge_improved" and artifact_dir.parent.name == "artifacts":
        return artifact_dir.parent.parent
    return Path.cwd().resolve()


def load_dataset_tables(project_root: Path) -> dict[str, pd.DataFrame]:
    data_root = project_root / "datasets" if (project_root / "datasets").exists() else project_root
    split_paths = {
        "Train": data_root / "RawCh_train" / "RawCh_train" / "balanced_index.csv",
        "Validation": data_root / "RawCh_val" / "RawCh_val" / "balanced_index.csv",
        "Test": data_root / "RawCh_test" / "RawCh_test" / "balanced_index.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for split, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset CSV for {split}: {path}")
        tables[split] = pd.read_csv(path)
    return tables


def save_current(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def mean_by_epoch(history: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        history.groupby("epoch", as_index=False)[column]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": f"{column}_mean", "std": f"{column}_std"})
    )


def plot_history_metric(
    history: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, (seed, group) in enumerate(history.groupby("seed")):
        ax.plot(
            group["epoch"],
            group[column],
            color=COLORS[idx % len(COLORS)],
            alpha=0.45,
            linewidth=1.8,
            label=f"Seed {seed}",
        )
    mean_df = mean_by_epoch(history, column)
    mean_col = f"{column}_mean"
    std_col = f"{column}_std"
    ax.plot(
        mean_df["epoch"],
        mean_df[mean_col],
        color="#111827",
        linewidth=3.2,
        label="Mean across seeds",
    )
    if not mean_df[std_col].isna().all():
        low = mean_df[mean_col] - mean_df[std_col].fillna(0)
        high = mean_df[mean_col] + mean_df[std_col].fillna(0)
        ax.fill_between(mean_df["epoch"], low, high, color="#94a3b8", alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(ncol=2)
    save_current(fig, out_path)


def plot_loss(history: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, (seed, group) in enumerate(history.groupby("seed")):
        color = COLORS[idx % len(COLORS)]
        ax.plot(group["epoch"], group["train_loss"], color=color, alpha=0.25, linestyle="--")
        ax.plot(group["epoch"], group["val_loss"], color=color, alpha=0.45, linewidth=1.8)
    train_mean = mean_by_epoch(history, "train_loss")
    val_mean = mean_by_epoch(history, "val_loss")
    ax.plot(
        train_mean["epoch"],
        train_mean["train_loss_mean"],
        color="#16a34a",
        linewidth=3,
        linestyle="--",
        label="Mean train loss",
    )
    ax.plot(
        val_mean["epoch"],
        val_mean["val_loss_mean"],
        color="#dc2626",
        linewidth=3,
        label="Mean validation loss",
    )
    ax.set_title("Loss per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.legend()
    save_current(fig, out_path)


def plot_validation_accuracy(seed_df: pd.DataFrame, summary: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ordered = seed_df.sort_values("seed")
    bars = ax.bar(
        ordered["seed"].astype(str),
        ordered["val_accuracy"],
        color=COLORS[: len(ordered)],
        edgecolor="#0f172a",
        linewidth=0.8,
    )
    ensemble_acc = summary["val_metrics"]["accuracy"]
    ax.axhline(
        ensemble_acc,
        color="#111827",
        linewidth=2.8,
        linestyle="--",
        label=f"Ensemble val accuracy: {ensemble_acc:.4f}",
    )
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in ordered["val_accuracy"]], padding=4)
    ax.set_ylim(max(0.0, ordered["val_accuracy"].min() - 0.04), 1.0)
    ax.set_title("Validation Accuracy by Seed")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Validation accuracy")
    ax.legend()
    save_current(fig, out_path)


def plot_per_class_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float, out_path: Path) -> None:
    y_pred = (y_score >= threshold).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    classes = ["Real", "Fake"]
    metrics = pd.DataFrame(
        {
            "Class": classes,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support,
        }
    )
    x = np.arange(len(classes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, metrics["Precision"], width, label="Precision", color="#2563eb")
    ax.bar(x, metrics["Recall"], width, label="Recall", color="#f97316")
    ax.bar(x + width, metrics["F1"], width, label="F1", color="#10b981")
    for i, support_value in enumerate(metrics["Support"]):
        ax.text(i, 0.04, f"n={support_value}", ha="center", va="center", color="#0f172a")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.06)
    ax.set_title("Per-Class Metrics on Test Set")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    save_current(fig, out_path)
    return metrics


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="#7c3aed", linewidth=3, label=f"ROC AUC = {roc_auc:.4f}")
    ax.fill_between(fpr, tpr, color="#c4b5fd", alpha=0.35)
    ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1.8, label="Random")
    ax.set_title("ROC Curve on Test Set")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    save_current(fig, out_path)
    return float(roc_auc)


def plot_confusion_matrix(y_true: np.ndarray, y_score: np.ndarray, threshold: float, out_path: Path) -> None:
    y_pred = (y_score >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_percent = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Pastel1", vmin=0, vmax=max(1, int(cm.max() * 1.8)))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion Matrix on Test Set (threshold={threshold:.3f})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticklabels(["Real", "Fake"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_percent[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color="#000000",
                fontsize=13,
                fontweight="bold",
            )
    save_current(fig, out_path)


def plot_dashboard(
    history: pd.DataFrame,
    seed_df: pd.DataFrame,
    summary: dict,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Publication Challenge Evaluation Dashboard", fontsize=20, fontweight="bold")

    mean_auc = mean_by_epoch(history, "val_auc")
    axes[0, 0].plot(mean_auc["epoch"], mean_auc["val_auc_mean"], color="#2563eb", linewidth=3)
    axes[0, 0].set_title("Validation ROC-AUC")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("AUC")

    train_mean = mean_by_epoch(history, "train_loss")
    val_mean = mean_by_epoch(history, "val_loss")
    axes[0, 1].plot(train_mean["epoch"], train_mean["train_loss_mean"], color="#16a34a", linewidth=2.8, label="Train")
    axes[0, 1].plot(val_mean["epoch"], val_mean["val_loss_mean"], color="#dc2626", linewidth=2.8, label="Val")
    axes[0, 1].set_title("Loss per Epoch")
    axes[0, 1].legend()

    mean_f1 = mean_by_epoch(history, "val_f1")
    axes[0, 2].plot(mean_f1["epoch"], mean_f1["val_f1_mean"], color="#f97316", linewidth=3)
    axes[0, 2].set_title("Validation F1")
    axes[0, 2].set_xlabel("Epoch")

    axes[1, 0].bar(seed_df["seed"].astype(str), seed_df["val_accuracy"], color=COLORS[: len(seed_df)])
    axes[1, 0].axhline(summary["val_metrics"]["accuracy"], color="#111827", linestyle="--", linewidth=2)
    axes[1, 0].set_title("Validation Accuracy by Seed")
    axes[1, 0].set_ylim(max(0.0, seed_df["val_accuracy"].min() - 0.04), 1.0)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    axes[1, 1].plot(fpr, tpr, color="#7c3aed", linewidth=3)
    axes[1, 1].fill_between(fpr, tpr, color="#c4b5fd", alpha=0.35)
    axes[1, 1].plot([0, 1], [0, 1], color="#64748b", linestyle="--")
    axes[1, 1].set_title(f"Test ROC Curve AUC={roc_auc:.4f}")

    cm = confusion_matrix(y_true, (y_score >= threshold).astype(np.int64), labels=[0, 1])
    axes[1, 2].imshow(cm, cmap="Pastel1", vmin=0, vmax=max(1, int(cm.max() * 1.8)))
    axes[1, 2].set_title("Confusion Matrix")
    axes[1, 2].set_xticks([0, 1])
    axes[1, 2].set_yticks([0, 1])
    axes[1, 2].set_xticklabels(["Real", "Fake"])
    axes[1, 2].set_yticklabels(["Real", "Fake"])
    for i in range(2):
        for j in range(2):
            axes[1, 2].text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="#000000",
            )

    for ax in axes.ravel():
        ax.grid(True, alpha=0.35)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_current(fig, out_path)


def write_metrics_table(metrics: pd.DataFrame, out_path: Path) -> None:
    metrics.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def plot_dataset_split_bar(dataset_tables: dict[str, pd.DataFrame], out_path: Path) -> pd.DataFrame:
    rows = [{"Split": split, "Samples": len(df)} for split, df in dataset_tables.items()]
    split_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(
        split_df["Split"],
        split_df["Samples"],
        color=["#2563eb", "#f97316", "#10b981"],
        edgecolor="#0f172a",
        linewidth=0.9,
    )
    ax.bar_label(bars, labels=[f"{v:,}" for v in split_df["Samples"]], padding=4, fontweight="bold")
    ax.set_title("Dataset Split Size")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of samples")
    ax.set_ylim(0, int(split_df["Samples"].max() * 1.18))
    save_current(fig, out_path)
    return split_df


def plot_class_distribution(dataset_tables: dict[str, pd.DataFrame], out_path: Path) -> pd.DataFrame:
    rows = []
    for split, df in dataset_tables.items():
        counts = df["label"].str.lower().value_counts()
        rows.append(
            {
                "Split": split,
                "Real": int(counts.get("real", 0)),
                "Fake": int(counts.get("fake", 0)),
            }
        )
    class_df = pd.DataFrame(rows)
    x = np.arange(len(class_df))
    fig, ax = plt.subplots(figsize=(10, 6))
    real_bars = ax.bar(x, class_df["Real"], label="Real", color="#38bdf8", edgecolor="#0f172a", linewidth=0.8)
    fake_bars = ax.bar(
        x,
        class_df["Fake"],
        bottom=class_df["Real"],
        label="Fake",
        color="#fb7185",
        edgecolor="#0f172a",
        linewidth=0.8,
    )
    for idx, row in class_df.iterrows():
        ax.text(idx, row["Real"] / 2, f"Real\n{row['Real']:,}", ha="center", va="center", fontweight="bold")
        ax.text(
            idx,
            row["Real"] + row["Fake"] / 2,
            f"Fake\n{row['Fake']:,}",
            ha="center",
            va="center",
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(class_df["Split"])
    ax.set_title("Class Distribution by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of samples")
    ax.legend()
    save_current(fig, out_path)
    return class_df


def plot_language_distribution(dataset_tables: dict[str, pd.DataFrame], out_path: Path) -> pd.DataFrame:
    rows = []
    for split, df in dataset_tables.items():
        counts = df["language"].str.lower().value_counts()
        rows.append(
            {
                "Split": split,
                "English": int(counts.get("en", 0)),
                "Hindi": int(counts.get("hi", 0)),
            }
        )
    language_df = pd.DataFrame(rows)
    x = np.arange(len(language_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    en_bars = ax.bar(
        x - width / 2,
        language_df["English"],
        width,
        label="English",
        color="#a78bfa",
        edgecolor="#0f172a",
        linewidth=0.8,
    )
    hi_bars = ax.bar(
        x + width / 2,
        language_df["Hindi"],
        width,
        label="Hindi",
        color="#facc15",
        edgecolor="#0f172a",
        linewidth=0.8,
    )
    ax.bar_label(en_bars, labels=[f"{v:,}" for v in language_df["English"]], padding=3, fontweight="bold")
    ax.bar_label(hi_bars, labels=[f"{v:,}" for v in language_df["Hindi"]], padding=3, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(language_df["Split"])
    ax.set_title("Language Distribution by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of samples")
    ax.legend()
    save_current(fig, out_path)
    return language_df


def add_pipeline_box(ax, xy: tuple[float, float], text: str, color: str, width: float = 1.7, height: float = 0.76) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        linewidth=1.6,
        edgecolor="#0f172a",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#0f172a",
        wrap=True,
    )


def add_pipeline_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="#334155",
    )
    ax.add_patch(arrow)


def plot_input_output_pipeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 5.8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.1)
    ax.axis("off")
    ax.set_title("Input-Output Pipeline for English-Hindi Deepfake Detection", pad=18)

    boxes = [
        ((0.25, 2.35), "Raw Audio\nEnglish / Hindi\nReal or Fake", "#dbeafe"),
        ((2.05, 2.35), "WavLM Feature\nExtraction", "#ede9fe"),
        ((3.85, 2.35), "768-D .npy\nEmbedding", "#cffafe"),
        ((5.65, 2.95), "Safari Branch\nAFUM + Transformer", "#dcfce7"),
        ((5.65, 1.75), "WavLM MLP\nBranch", "#fef9c3"),
        ((5.65, 0.55), "Language\nEmbedding", "#ffedd5"),
        ((7.65, 1.75), "Fusion\nClassifier", "#fae8ff"),
        ((9.25, 1.75), "Output\nFake Probability\nReal / Fake", "#fee2e2"),
    ]
    for xy, text, color in boxes:
        add_pipeline_box(ax, xy, text, color)

    add_pipeline_arrow(ax, (1.95, 2.73), (2.05, 2.73))
    add_pipeline_arrow(ax, (3.75, 2.73), (3.85, 2.73))
    add_pipeline_arrow(ax, (5.55, 2.73), (5.65, 3.33))
    add_pipeline_arrow(ax, (5.55, 2.73), (5.65, 2.13))
    add_pipeline_arrow(ax, (5.55, 2.73), (5.65, 0.93))
    add_pipeline_arrow(ax, (7.35, 3.33), (7.65, 2.35))
    add_pipeline_arrow(ax, (7.35, 2.13), (7.65, 2.13))
    add_pipeline_arrow(ax, (7.35, 0.93), (7.65, 1.92))
    add_pipeline_arrow(ax, (9.35, 2.13), (9.25, 2.13))

    ax.text(
        5.0,
        0.12,
        "Final experiment: RawCh held-out-generator split + WavLM embeddings + 5-seed SafariWavLMEnsemble.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#334155",
        fontweight="bold",
    )
    save_current(fig, out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate colorful evaluation graphs for the publication challenge model."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/publication_challenge_improved"),
        help="Directory containing publication_challenge_summary.json and seed_* artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for generated graphs. Defaults to <artifact-dir>/graphs.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    artifact_dir = args.artifact_dir.resolve()
    out_dir = (args.out_dir or artifact_dir / "graphs").resolve()
    summary_path = artifact_dir / "publication_challenge_summary.json"
    predictions_path = artifact_dir / "ensemble_predictions" / "test_improved_predictions.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing test predictions: {predictions_path}")

    set_style()
    summary = load_json(summary_path)
    project_root = infer_project_root(artifact_dir)
    dataset_tables = load_dataset_tables(project_root)
    seed_df, history = load_seed_metrics(artifact_dir)
    if seed_df.empty:
        raise RuntimeError(f"No seed metrics found under {artifact_dir}")
    if history.empty:
        raise RuntimeError(f"No history rows found under {artifact_dir}")

    y_true, y_score = load_predictions(predictions_path)
    threshold = float(summary["threshold"])

    plot_history_metric(
        history,
        "val_auc",
        "Validation ROC-AUC per Epoch",
        "ROC-AUC",
        out_dir / "01_validation_roc_auc_per_epoch.png",
        ylim=(0.6, 1.01),
    )
    plot_loss(history, out_dir / "02_loss_per_epoch.png")
    plot_history_metric(
        history,
        "val_f1",
        "Validation F1 Score per Epoch",
        "F1 score",
        out_dir / "03_validation_f1_per_epoch.png",
        ylim=(0.45, 1.02),
    )
    plot_validation_accuracy(seed_df, summary, out_dir / "04_validation_accuracy_by_seed.png")
    per_class_df = plot_per_class_metrics(
        y_true,
        y_score,
        threshold,
        out_dir / "05_per_class_metrics_test_set.png",
    )
    plot_roc_curve(y_true, y_score, out_dir / "06_roc_curve_test_set.png")
    plot_confusion_matrix(y_true, y_score, threshold, out_dir / "07_confusion_matrix_test_set.png")
    plot_dashboard(
        history,
        seed_df,
        summary,
        y_true,
        y_score,
        threshold,
        out_dir / "08_evaluation_dashboard.png",
    )
    split_df = plot_dataset_split_bar(dataset_tables, out_dir / "09_dataset_split_bar_chart.png")
    class_df = plot_class_distribution(dataset_tables, out_dir / "10_class_distribution_chart.png")
    language_df = plot_language_distribution(dataset_tables, out_dir / "11_language_distribution_chart.png")
    plot_input_output_pipeline(out_dir / "12_input_output_pipeline_figure.png")

    write_metrics_table(seed_df, out_dir / "seed_metrics_table.csv")
    write_metrics_table(per_class_df, out_dir / "per_class_metrics_table.csv")
    write_metrics_table(split_df, out_dir / "dataset_split_table.csv")
    write_metrics_table(class_df, out_dir / "class_distribution_table.csv")
    write_metrics_table(language_df, out_dir / "language_distribution_table.csv")

    print("\nGraph generation complete.")
    print(f"Artifact directory: {artifact_dir}")
    print(f"Output directory:   {out_dir}")
    print(f"Threshold:          {threshold:.4f}")
    print(f"Test accuracy:      {accuracy_score(y_true, (y_score >= threshold).astype(np.int64)):.4f}")


if __name__ == "__main__":
    main()
