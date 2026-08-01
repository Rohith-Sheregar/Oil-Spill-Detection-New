"""
Module 1 training report generator.

Produces:
  results/module1/report/module1_report.html  — full HTML report with embedded plots
  results/module1/plots/*.png                 — individual PNG figures

Figures generated
-----------------
1.  training_curves.png       — train/val loss curves + val mIoU over epochs
2.  lr_schedule.png           — learning-rate schedule (cosine annealing)
3.  class_distribution.png    — pie chart of oil/lookalike/no-oil scene counts
4.  band_stats.png            — per-band mean & std of the 5 bands
5.  sample_predictions.png    — grid of sample image / prediction / ground truth
6.  confusion_heatmap.png     — val-set pixel-level confusion matrix
7.  iou_per_class.png         — IoU per class per epoch
8.  pseudo_label_history.png  — pseudo-label cycle progression (if run)
"""
from __future__ import annotations

import base64
import json
import logging
import textwrap
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

log = logging.getLogger(__name__)


# ─── Style ────────────────────────────────────────────────────────────────────

BRAND_BLUE  = "#2D7DD2"
BRAND_RED   = "#E84855"
BRAND_GREEN = "#3BB273"
BRAND_ORANGE = "#EF8C2E"
BRAND_PURPLE = "#8B5CF6"

plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})


# ─── PNG helpers ──────────────────────────────────────────────────────────────

def _fig_to_png(fig: plt.Figure, plots_dir: Path, filename: str) -> str:
    """Save figure to disk AND return base64-encoded PNG data-URI for HTML embedding."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / filename
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}", str(out_path)


# ─── Individual plot functions ────────────────────────────────────────────────

def plot_training_curves(history: dict, plots_dir: Path) -> tuple[str, str]:
    """Plot loss and mIoU training curves."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Module 1 — Training Curves", fontweight="bold")

    ax1.plot(epochs, history["train_loss"], color=BRAND_BLUE,   label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   color=BRAND_RED,    label="Val Loss",   linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE+Dice Loss"); ax1.set_title("Loss")
    ax1.legend()

    if "val_miou" in history:
        ax2.plot(epochs, history["val_miou"], color=BRAND_GREEN, label="Val mIoU")
        if "val_f1" in history:
            ax2.plot(epochs, history["val_f1"], color=BRAND_ORANGE, label="Val F1", linestyle="--")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score"); ax2.set_title("Validation Metrics")
        ax2.legend()
        ax2.set_ylim(0, 1)

    plt.tight_layout()
    return _fig_to_png(fig, plots_dir, "training_curves.png")


def plot_lr_schedule(history: dict, plots_dir: Path) -> tuple[str, str]:
    """Plot learning-rate schedule if recorded."""
    if "lr" not in history:
        return None, None
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history["lr"], color=BRAND_PURPLE)
    ax.set_xlabel("Step"); ax.set_ylabel("Learning Rate")
    ax.set_title("Module 1 — Learning Rate Schedule (CosineAnnealingWarmRestarts)")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.tight_layout()
    return _fig_to_png(fig, plots_dir, "lr_schedule.png")


def plot_class_distribution(metadata: dict, plots_dir: Path) -> tuple[str, str]:
    """Pie chart of scene class distribution in the training set."""
    counts = metadata.get("class_counts", {})
    if not counts:
        return None, None
    labels = list(counts.keys())
    sizes  = list(counts.values())
    colors = [BRAND_RED, BRAND_ORANGE, BRAND_BLUE]
    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors[:len(labels)],
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax.set_title("Training Scene Class Distribution", fontweight="bold", pad=15)
    plt.tight_layout()
    return _fig_to_png(fig, plots_dir, "class_distribution.png")


def plot_band_stats(band_stats: list[dict], plots_dir: Path) -> tuple[str, str]:
    """Per-band mean ± std bar chart for the 5-band stack."""
    if not band_stats:
        return None, None
    names  = [b["name"] for b in band_stats]
    means  = [b["mean"] for b in band_stats]
    stds   = [b["std"]  for b in band_stats]
    colors = [BRAND_BLUE, BRAND_RED, BRAND_GREEN, BRAND_ORANGE, BRAND_PURPLE]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, means, width=0.5, color=colors[:len(names)], yerr=stds,
                  capsize=5, error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Band value (normalised)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Feature Band Statistics (train set, normalised [0, 1])", fontweight="bold")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    return _fig_to_png(fig, plots_dir, "band_stats.png")


def plot_sample_predictions(
    samples: list[dict],   # each: {"image": (5,H,W) np, "mask": (H,W) np, "pred": (H,W) np}
    plots_dir: Path,
    n_cols: int = 4,
) -> tuple[str, str]:
    """Grid of (image VV | image VH | prediction | ground truth) for sample patches."""
    if not samples:
        return None, None
    n = min(len(samples), n_cols)
    fig = plt.figure(figsize=(4 * n, 16))
    gs  = gridspec.GridSpec(4, n, hspace=0.05, wspace=0.05)

    def pstretch(a):
        lo, hi = np.percentile(a, [2, 98])
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

    row_titles = ["VV band", "VH band", "Prediction", "Ground Truth"]
    for col_i, sample in enumerate(samples[:n]):
        img  = sample["image"]   # (5, H, W)
        mask = sample["mask"]    # (H, W)
        pred = sample["pred"]    # (H, W)
        for row_i, (data, cmap) in enumerate([
            (pstretch(img[0]), "gray"),
            (pstretch(img[1]), "gray"),
            (pred,             "RdYlGn_r"),
            (mask,             "RdYlGn_r"),
        ]):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            if col_i == 0:
                ax.set_ylabel(row_titles[row_i], fontsize=10, labelpad=2)

    fig.suptitle("Module 1 — Sample Predictions (val set)", fontweight="bold", fontsize=13)
    return _fig_to_png(fig, plots_dir, "sample_predictions.png")


def plot_confusion_heatmap(cm: np.ndarray, plots_dir: Path) -> tuple[str, str]:
    """2×2 pixel-level confusion matrix as annotated heatmap."""
    if cm is None:
        return None, None
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    labels = ["Background", "Oil Spill"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Pixel-level Confusion Matrix (val set)", fontweight="bold")
    total = cm.sum() + 1e-9
    for (r, c), v in np.ndenumerate(cm):
        pct = 100 * v / total
        ax.text(c, r, f"{v:,}\n({pct:.1f}%)", ha="center", va="center",
                fontsize=9, color="white" if v > cm.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return _fig_to_png(fig, plots_dir, "confusion_heatmap.png")


def plot_pseudo_history(pseudo_history: list[dict], plots_dir: Path) -> tuple[str, str]:
    """Pseudo-label cycle progression chart."""
    if not pseudo_history:
        return None, None
    cycles = [r["cycle"] for r in pseudo_history]
    n_pseudo = [r["n_pseudo_scenes"] for r in pseudo_history]
    losses   = [r["mean_epoch_loss"]  for r in pseudo_history]
    thresholds = [r["conf_threshold"]  for r in pseudo_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Pseudo-Label Self-Evolution Cycles", fontweight="bold")

    ax1.bar(cycles, n_pseudo, color=BRAND_BLUE, alpha=0.8)
    ax1b = ax1.twinx()
    ax1b.plot(cycles, thresholds, "o--", color=BRAND_RED, label="Conf threshold")
    ax1.set_xlabel("Cycle"); ax1.set_ylabel("Pseudo scenes accepted", color=BRAND_BLUE)
    ax1b.set_ylabel("Confidence threshold", color=BRAND_RED)
    ax1.set_title("Accepted Scenes vs Cycle")

    ax2.plot(cycles, losses, "s-", color=BRAND_GREEN)
    ax2.set_xlabel("Cycle"); ax2.set_ylabel("Mean epoch loss")
    ax2.set_title("Fine-tune Loss per Cycle")

    plt.tight_layout()
    return _fig_to_png(fig, plots_dir, "pseudo_label_history.png")


# ─── HTML builder ─────────────────────────────────────────────────────────────

def _img_html(data_uri: str, caption: str, width: str = "100%") -> str:
    return (
        f'<figure style="margin:20px 0">'
        f'<img src="{data_uri}" style="width:{width};border-radius:8px;'
        f'box-shadow:0 4px 12px rgba(0,0,0,0.1)" alt="{caption}">'
        f'<figcaption style="text-align:center;color:#555;margin-top:6px">{caption}</figcaption>'
        f'</figure>'
    )


def build_html_report(
    report_dir:  Path,
    plots_dir:   Path,
    history:     dict,
    metadata:    dict,
    band_stats:  list[dict] | None = None,
    samples:     list[dict] | None = None,
    cm:          np.ndarray | None = None,
    pseudo_history: list[dict] | None = None,
) -> Path:
    """
    Generate the full HTML report.

    Parameters
    ----------
    report_dir      : directory where module1_report.html is written
    plots_dir       : directory for individual PNG files
    history         : {"train_loss": [...], "val_loss": [...], "val_miou": [...], ...}
    metadata        : {"class_counts": {...}, "n_train": int, "n_val": int, ...}
    band_stats      : [{"name": str, "mean": float, "std": float}, ...]
    samples         : list of {"image", "mask", "pred"} dicts
    cm              : 2×2 np.ndarray confusion matrix
    pseudo_history  : output of run_pseudo_label_cycles()
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate all plots ────────────────────────────────────────────────
    plots = {}
    if "train_loss" in history:
        uri, path = plot_training_curves(history, plots_dir)
        plots["training_curves"] = (uri, "Figure 1: Training and Validation Curves")

    if "lr" in history:
        uri, path = plot_lr_schedule(history, plots_dir)
        if uri:
            plots["lr_schedule"] = (uri, "Figure 2: Learning Rate Schedule")

    if metadata.get("class_counts"):
        uri, path = plot_class_distribution(metadata, plots_dir)
        if uri:
            plots["class_dist"] = (uri, "Figure 3: Training Set Class Distribution")

    if band_stats:
        uri, path = plot_band_stats(band_stats, plots_dir)
        if uri:
            plots["band_stats"] = (uri, "Figure 4: Feature Band Statistics")

    if samples:
        uri, path = plot_sample_predictions(samples, plots_dir)
        if uri:
            plots["predictions"] = (uri, "Figure 5: Sample Predictions (val set)")

    if cm is not None:
        uri, path = plot_confusion_heatmap(cm, plots_dir)
        if uri:
            plots["confusion"] = (uri, "Figure 6: Pixel-level Confusion Matrix")

    if pseudo_history:
        uri, path = plot_pseudo_history(pseudo_history, plots_dir)
        if uri:
            plots["pseudo"] = (uri, "Figure 7: Pseudo-label Self-Evolution")

    # ── Build metric summary table ─────────────────────────────────────────
    best_epoch = int(np.argmin(history.get("val_loss", [0]))) + 1
    best_val_loss = min(history.get("val_loss", [float("nan")]))
    best_miou     = max(history.get("val_miou", [float("nan")]))
    best_f1       = max(history.get("val_f1",   [float("nan")]))

    def _metric_row(name, value, unit=""):
        val_str = f"{value:.4f}{unit}" if isinstance(value, float) else str(value)
        return f"<tr><td><b>{name}</b></td><td>{val_str}</td></tr>"

    metric_rows = "\n".join([
        _metric_row("Best epoch",              best_epoch),
        _metric_row("Best val BCE+Dice loss",  best_val_loss),
        _metric_row("Best val mIoU",           best_miou),
        _metric_row("Best val F1",             best_f1),
        _metric_row("Total epochs",            len(history.get("train_loss", []))),
        _metric_row("Train scenes",            metadata.get("n_train", "—")),
        _metric_row("Val scenes",              metadata.get("n_val", "—")),
        _metric_row("Input mode",              metadata.get("input_mode", "full_5band")),
        _metric_row("Patch size",              metadata.get("patch_size", 256)),
        _metric_row("Model",                   "DeepLabV3+ / MobileNetV2 + scSE"),
        _metric_row("Loss",                    "BCE(ε=0.1) + Dice(smooth=1)"),
        _metric_row("Generated",               datetime.now().strftime("%Y-%m-%d %H:%M")),
    ])

    # ── Build plot section ─────────────────────────────────────────────────
    plot_html = "\n".join(
        _img_html(uri, caption) for (uri, caption) in plots.values()
    )

    # ── Full HTML ──────────────────────────────────────────────────────────
    html = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Module 1 — Oil Spill Segmentation Report</title>
      <style>
        *     {{ box-sizing: border-box; margin: 0; padding: 0 }}
        body  {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f8fafc; color: #1e293b }}
        .wrap {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px }}
        h1    {{ font-size: 1.9rem; color: {BRAND_BLUE}; border-bottom: 3px solid {BRAND_BLUE}; padding-bottom: 10px; margin-bottom: 24px }}
        h2    {{ font-size: 1.3rem; margin: 32px 0 14px; color: #334155 }}
        .card {{ background: #fff; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.07); padding: 24px; margin-bottom: 28px }}
        table {{ border-collapse: collapse; width: 100% }}
        td    {{ padding: 8px 14px; border-bottom: 1px solid #e2e8f0 }}
        tr:last-child td {{ border-bottom: none }}
        tr:nth-child(even) {{ background: #f1f5f9 }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: .85rem; font-weight: 600 }}
        .blue  {{ background: #dbeafe; color: #1e40af }}
        .green {{ background: #dcfce7; color: #166534 }}
        footer {{ text-align: center; margin-top: 40px; color: #94a3b8; font-size: .85rem }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <h1>&#9732; Module 1 &mdash; SAR Preprocessing &amp; Oil Spill Segmentation Report</h1>
        <p style="color:#64748b;margin-bottom:20px">
          Dataset: Sentinel-1 SAR Oil Spill Dataset (Trujillo-Acatitla et al. 2024, Zenodo)
          &nbsp;&bull;&nbsp; Architecture: DeepLabV3+ / MobileNetV2 + scSE
          &nbsp;&bull;&nbsp; Input: 5-band stack (VV, VH, Entropy H, Alpha, Wind-ratio)
        </p>

        <div class="card">
          <h2>&#127358; Key Metrics</h2>
          <table>
            <thead><tr style="background:#2D7DD2;color:#fff"><th style="padding:10px 14px">Metric</th><th style="padding:10px 14px">Value</th></tr></thead>
            <tbody>{metric_rows}</tbody>
          </table>
        </div>

        <div class="card">
          <h2>&#128202; Training Plots</h2>
          {plot_html}
        </div>

        <div class="card">
          <h2>&#128221; Method Notes</h2>
          <ul style="line-height:1.8;padding-left:20px">
            <li><b>Bands 0-1 (VV, VH):</b> Sigma0 calibrated backscatter in dB, robust-percentile normalised to [0,1].</li>
            <li><b>Band 2 (Entropy H):</b> Dual-pol Cloude-Pottier Shannon entropy. H≈0 = single dominant scattering mechanism (specular/calm sea); H≈1 = depolarising medium (ocean roughness, oil trapping).</li>
            <li><b>Band 3 (RVI_dp):</b> Dual-pol Radar Vegetation Index, 4·VH/(VV+VH). Substitutes the synopsis's Cloude-Pottier alpha angle, which requires complex SLC phase unavailable in Sentinel-1 GRD amplitude products. Low values indicate depolarisation-suppressed, smooth (potentially oil-dampened) surfaces; higher values indicate rougher, more depolarising ocean backscatter.</li>
            <li><b>Band 4 (Wind-corrected ratio):</b> Raw VV/VH ratio divided by the CMOD5.N prediction for the scene wind speed. Removes wind-driven backscatter variation so oil signature is wind-invariant.</li>
            <li><b>scSE attention:</b> Channel squeeze-and-excitation + spatial squeeze-and-excitation applied to the decoder output. Focuses the network on dark oil patches against bright sea clutter.</li>
            <li><b>Loss:</b> BCE(ε=0.1) + Dice(smooth=1). Label smoothing prevents overconfident predictions on ambiguous oil boundaries. Dice handles severe class imbalance (&lt;1% positive pixels per scene).</li>
            <li><b>Pseudo-labelling:</b> Up to 10 self-evolution cycles on the unlabelled lookalike/no-oil scenes. Confidence threshold raised 5 pp per cycle (0.70 → 0.92).</li>
          </ul>
        </div>

        <footer>Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} &bull; Oil Spill Detection — Module 1</footer>
      </div>
    </body>
    </html>
    """).strip()

    out_path = report_dir / "module1_report.html"
    out_path.write_text(html, encoding="utf-8")
    log.info("HTML report written: %s", out_path)
    return out_path


# ─── Public convenience wrapper ───────────────────────────────────────────────

def generate_module1_report(
    results_dir: Path | str,
    history: dict,
    metadata: dict,
    band_stats: list[dict] | None = None,
    samples: list[dict] | None = None,
    cm: np.ndarray | None = None,
    pseudo_history: list[dict] | None = None,
) -> Path:
    """
    One-call entry point. Call this at the end of training.

    Parameters
    ----------
    results_dir : Path to the module1 results directory (e.g. results/module1/)
    history     : training history dict from the training loop
    metadata    : run metadata dict (class_counts, n_train, n_val, ...)
    band_stats  : optional list of per-band statistics
    samples     : optional list of sample prediction dicts for visualisation
    cm          : optional 2×2 pixel confusion matrix
    pseudo_history : optional pseudo-label cycle history

    Returns
    -------
    Path to the generated HTML report
    """
    results_dir = Path(results_dir)
    plots_dir   = results_dir / "plots"
    report_dir  = results_dir / "report"
    return build_html_report(
        report_dir, plots_dir, history, metadata,
        band_stats=band_stats, samples=samples,
        cm=cm, pseudo_history=pseudo_history,
    )
