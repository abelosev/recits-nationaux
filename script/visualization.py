#!/usr/bin/env python3
"""
Génère 4–5 figures (PNG) à partir des sorties de step6_cluster.py.

Usage:
  python3 visualization.py ../clustering_out ../figures --top-k 20 --prob-th 0.8

"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv(clustering_dir: Path) -> pd.DataFrame:
    csv_path = clustering_dir / "segments_with_clusters.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)

    for col in ["cluster", "idx"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "probability" in df.columns:
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    required = ["x_2d", "y_2d", "cluster", "period"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante dans le CSV: '{c}'")

    if "is_noise" not in df.columns:
        df["is_noise"] = (df["cluster"] == -1)
    else:
        if df["is_noise"].dtype == object:
            df["is_noise"] = df["is_noise"].astype(str).str.lower().isin(["true", "1", "yes"])

    df["cluster"] = df["cluster"].fillna(-1).astype(int)
    df["period"] = df["period"].fillna("unknown").astype(str)

    return df


def maybe_read_stats(clustering_dir: Path) -> dict:
    stats_path = clustering_dir / "clustering_stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def savefig(path: Path, dpi: int = 220) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def get_cluster_color_map(cluster_ids):
    """
    Construit un mapping cluster_id -> couleur
    """
    cluster_ids = [c for c in cluster_ids if c != -1]
    cluster_ids = sorted(set(cluster_ids))

    tab = plt.get_cmap("tab20")
    hsv = plt.get_cmap("hsv")

    cmap = {}
    for i, cid in enumerate(cluster_ids):
        if i < 20:
            cmap[cid] = tab(i)
        else:
            cmap[cid] = hsv((i - 20) / max(1, (len(cluster_ids) - 20)))
    return cmap

# Figures
def fig_umap_clusters_core(df: pd.DataFrame, outdir: Path, prob_th: float, top_k: int) -> None:
    """
    UMAP 2D: on montre surtout les clusters les plus gros (top_k) et
    uniquement les points "core" (prob>=prob_th) pour éviter la bouillie.
    Bruit en gris clair.
    """
    clustered = df[~df["is_noise"]].copy()
    if "probability" in clustered.columns:
        clustered = clustered[clustered["probability"].fillna(0.0) >= prob_th]

    sizes = clustered["cluster"].value_counts().sort_values(ascending=False)
    keep_clusters = list(sizes.head(top_k).index)
    clustered = clustered[clustered["cluster"].isin(keep_clusters)]

    noise = df[df["is_noise"]]

    cmap = get_cluster_color_map(keep_clusters)

    plt.figure(figsize=(9, 7))
    plt.scatter(noise["x_2d"], noise["y_2d"], s=8, alpha=0.25, linewidths=0)

    for cid in keep_clusters:
        part = clustered[clustered["cluster"] == cid]
        if part.empty:
            continue
        plt.scatter(part["x_2d"], part["y_2d"], s=10, alpha=0.75, linewidths=0, label=f"C{cid}", c=[cmap[cid]])

    plt.title(f"UMAP 2D — points \"core\" (prob ≥ {prob_th}) — top {top_k} clusters")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(ncol=4, fontsize=8, frameon=False, loc="upper right")
    savefig(outdir / "umap_clusters_core.png")


def fig_umap_periods(df: pd.DataFrame, outdir: Path) -> None:
    """
    UMAP 2D coloré par période. Bruit en gris.
    """
    periods = sorted(df["period"].unique())
    tab = plt.get_cmap("tab10")
    hsv = plt.get_cmap("hsv")

    period_color = {}
    for i, p in enumerate(periods):
        if i < 10:
            period_color[p] = tab(i)
        else:
            period_color[p] = hsv(i / max(1, len(periods)))

    noise = df[df["is_noise"]]
    clustered = df[~df["is_noise"]]

    plt.figure(figsize=(9, 7))
    plt.scatter(noise["x_2d"], noise["y_2d"], s=8, alpha=0.20, linewidths=0)

    for p in periods:
        part = clustered[clustered["period"] == p]
        if part.empty:
            continue
        plt.scatter(part["x_2d"], part["y_2d"], s=10, alpha=0.70, linewidths=0, label=p, c=[period_color[p]])

    plt.title("UMAP 2D — coloration par période (bruit en gris)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(fontsize=9, frameon=False, loc="best")
    savefig(outdir / "umap_periods.png")


def fig_probability_hist(df: pd.DataFrame, outdir: Path) -> None:
    """
    Histogramme des probabilities pour les points clusterisés.
    """
    if "probability" not in df.columns:
        print("Colonne 'probability' absente du CSV: skip prob_hist.png")
        return

    clustered = df[~df["is_noise"]].copy()
    probs = clustered["probability"].dropna().values
    if probs.size == 0:
        print("Aucune probability disponible: skip prob_hist.png")
        return

    plt.figure(figsize=(8, 4.8))
    plt.hist(probs, bins=30)
    plt.title("Distribution des probabilités HDBSCAN (points clusterisés)")
    plt.xlabel("Probability")
    plt.ylabel("Nombre de segments")
    savefig(outdir / "prob_hist.png")


def fig_cluster_sizes_top(df: pd.DataFrame, outdir: Path, top_k: int) -> None:
    """
    Barplot top clusters par taille (hors bruit).
    """
    clustered = df[~df["is_noise"]]
    sizes = clustered["cluster"].value_counts().sort_values(ascending=False).head(top_k)

    plt.figure(figsize=(10, 5))
    plt.bar([str(i) for i in sizes.index], sizes.values)
    plt.title(f"Tailles des clusters (top {top_k}, hors bruit)")
    plt.xlabel("Cluster ID")
    plt.ylabel("Nombre de segments")
    savefig(outdir / "cluster_sizes_top.png")


def fig_heatmap_period_cluster(df: pd.DataFrame, outdir: Path, top_k: int) -> None:
    """
    Heatmap période × cluster (% dans la période), sur top_k clusters globaux.
    """
    clustered = df[~df["is_noise"]].copy()

    top_clusters = list(clustered["cluster"].value_counts().head(top_k).index)
    clustered = clustered[clustered["cluster"].isin(top_clusters)]

    pivot = pd.pivot_table(
        clustered,
        index="period",
        columns="cluster",
        values="idx",
        aggfunc="count",
        fill_value=0
    )

    pivot_pct = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100
    pivot_pct = pivot_pct.fillna(0)

    pivot_pct = pivot_pct.reindex(sorted(pivot_pct.columns), axis=1)

    plt.figure(figsize=(12, 5.5))
    im = plt.imshow(pivot_pct.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="% dans la période")

    plt.yticks(range(len(pivot_pct.index)), pivot_pct.index)
    plt.xticks(range(len(pivot_pct.columns)), [str(c) for c in pivot_pct.columns], rotation=90)
    plt.title(f"Heatmap: répartition des clusters par période (top {top_k})")
    plt.xlabel("Cluster ID")
    plt.ylabel("Période")
    savefig(outdir / "heatmap_period_cluster.png")


def main():
    parser = argparse.ArgumentParser(description="Génère 4–5 figures à partir de clustering_out (step6).")
    parser.add_argument("clustering_dir", type=Path, help="Dossier de sortie step6 (contient segments_with_clusters.csv).")
    parser.add_argument("outdir", type=Path, help="Dossier de sortie des figures (PNG).")
    parser.add_argument("--top-k", type=int, default=20, help="Nombre de clusters à afficher (top K). Défaut: 20")
    parser.add_argument("--prob-th", type=float, default=0.8, help="Seuil probability pour 'core'. Défaut: 0.8")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    df = read_csv(args.clustering_dir)
    _stats = maybe_read_stats(args.clustering_dir)

    # Figures
    fig_umap_clusters_core(df, args.outdir, prob_th=args.prob_th, top_k=args.top_k)
    fig_umap_periods(df, args.outdir)
    fig_probability_hist(df, args.outdir)
    fig_cluster_sizes_top(df, args.outdir, top_k=args.top_k)
    fig_heatmap_period_cluster(df, args.outdir, top_k=args.top_k)

if __name__ == "__main__":
    main()