"""Static PCA cluster plot of the real command-sample embeddings.

Reads the JSON written by export_command_embeddings.py (real 768-dim
Whisper encoder embeddings for the 31 recordings in storage/voice_samples/)
and draws a 2D scatter, one color per command, with a KMeans(k=6) cluster
boundary overlay so you can see how well unsupervised clustering recovers
the known command labels.

Usage (from backend/, with the venv active):
    python -m scripts.plot_command_clusters ../command_embeddings.json ../command_clusters.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans

COLORS = {
    "delete": "#2a78d6",
    "next": "#eb6834",
    "previous": "#1baf7a",
    "save": "#eda100",
    "stop": "#e87ba4",
    "submit": "#008300",
}
ORDER = ["delete", "next", "previous", "save", "stop", "submit"]


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../command_embeddings.json")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("../command_clusters.png")

    data = json.loads(in_path.read_text())
    points = data["points"]
    xy = np.array([p["pca2"] for p in points])
    labels = [p["command_id"] for p in points]

    kmeans = KMeans(n_clusters=len(ORDER), n_init=10, random_state=0).fit(xy)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    # Voronoi-ish background wash for the k-means partitions.
    x_min, x_max = xy[:, 0].min() - 0.08, xy[:, 0].max() + 0.08
    y_min, y_max = xy[:, 1].min() - 0.08, xy[:, 1].max() + 0.08
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )
    grid_labels = kmeans.predict(np.c_[grid_x.ravel(), grid_y.ravel()]).reshape(grid_x.shape)
    ax.contourf(grid_x, grid_y, grid_labels, levels=len(ORDER), colors="#e1e0d9", alpha=0.35)
    ax.contour(grid_x, grid_y, grid_labels, levels=len(ORDER), colors="#c3c2b7", linewidths=0.8)

    for command in ORDER:
        idx = [i for i, l in enumerate(labels) if l == command]
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=70,
            color=COLORS[command],
            edgecolor="#fcfcfb",
            linewidth=1.2,
            label=command,
            zorder=3,
        )

    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="x",
        s=90,
        color="#0b0b0b",
        linewidth=1.6,
        zorder=4,
        label="k-means centroid",
    )

    var = data["explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)", fontsize=10, color="#52514e")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)", fontsize=10, color="#52514e")
    ax.set_title(
        f"Voice-command embedding clusters ({len(points)} real clips, {len(ORDER)} commands)",
        fontsize=13,
        fontweight="bold",
        color="#0b0b0b",
        pad=14,
    )
    ax.tick_params(colors="#898781", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[c], markersize=8, label=c)
        for c in ORDER
    ]
    handles.append(Line2D([0], [0], marker="x", color="#0b0b0b", markersize=8, linestyle="none", label="k-means centroid"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
