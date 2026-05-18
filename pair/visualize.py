from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .algorithm import RunResult

_COLORS = ["#2196F3", "#4CAF50", "#FF9800"]


def plot_benchmark(results: list[RunResult], output_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PAIR Benchmark: N×K Configuration Comparison", fontsize=14, fontweight="bold")

    labels = [r.config.label for r in results]

    # --- Plot 1: Score distribution ---
    ax = axes[0]
    x = np.arange(1, 11)
    bar_width = 0.25
    for i, result in enumerate(results):
        counts = [0] * 11
        for stream in result.streams:
            counts[stream.best_score] += 1
        ax.bar(x + i * bar_width, counts[1:], bar_width, label=labels[i], color=_COLORS[i], alpha=0.85)
    ax.set_xlabel("Best Score (1–10)")
    ax.set_ylabel("Number of Streams")
    ax.set_title("Score Distribution per Config")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([str(s) for s in range(1, 11)])
    ax.legend()

    # --- Plot 2: Per-stream scores (sorted descending) ---
    ax = axes[1]
    for i, result in enumerate(results):
        scores = sorted([s.best_score for s in result.streams], reverse=True)
        ax.plot(range(1, len(scores) + 1), scores, marker="o", markersize=4,
                label=labels[i], color=_COLORS[i], linewidth=2)
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="Success (10)")
    ax.set_xlabel("Stream (sorted by score)")
    ax.set_ylabel("Best Score")
    ax.set_title("Per-Stream Scores (Sorted)")
    ax.set_ylim(0, 11)
    ax.legend()

    # --- Plot 3: Efficiency (total API calls vs best score) ---
    ax = axes[2]
    for i, result in enumerate(results):
        ax.scatter(result.total_api_calls, result.best_score,
                   s=220, color=_COLORS[i], label=labels[i], zorder=5)
        ax.annotate(
            f"{labels[i]}\n{result.elapsed_seconds:.1f}s",
            (result.total_api_calls, result.best_score),
            textcoords="offset points", xytext=(8, 4), fontsize=9,
        )
    ax.set_xlabel("Total API Calls")
    ax.set_ylabel("Best Score Achieved")
    ax.set_title("Efficiency: API Calls vs Best Score")
    ax.set_ylim(0, 11)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved to: {output_path}")
