"""
Étape 6 : Clusterisation avec HDBSCAN
Regroupe les embeddings pour faire émerger des thématiques narratives.

Pipeline: embeddings → UMAP (réduction) → HDBSCAN (clustering)
"""

import csv
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings('once', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module=r'umap(\..*)?$')
warnings.filterwarnings('ignore', category=UserWarning, module=r'numba(\..*)?$')
warnings.filterwarnings('ignore', category=DeprecationWarning, module=r'sklearn(\..*)?$')


def load_embeddings(embeddings_dir: Path, prefix: str = "all") -> Tuple[np.ndarray, List[Dict], List[str]]:
    vectors_path = embeddings_dir / f"{prefix}_vectors.npy"
    metadata_path = embeddings_dir / f"{prefix}_metadata.json"
    texts_path = embeddings_dir / f"{prefix}_texts.json"
    
    vectors = np.load(vectors_path)
    print(f"  Vecteurs: {vectors.shape}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts_data = json.load(f)
        texts = [t['text'] for t in texts_data]
    
    print(f"  Segments: {len(texts)}")
    
    if not (vectors.shape[0] == len(metadata) == len(texts)):
        print(f"ERREUR: Incohérence des tailles!")
        print(f"  vectors: {vectors.shape[0]}")
        print(f"  metadata: {len(metadata)}")
        print(f"  texts: {len(texts)}")
        sys.exit(1)
    
    return vectors, metadata, texts


# UMAP paramètres

def reduce_dimensions(
    embeddings: np.ndarray, 
    n_components: int = 50,
    n_components_viz: int = 2,
    method: str = 'umap',
    random_state: int = 42,
    umap_n_neighbors: int = 15,
    umap_min_dist_cluster: float = 0.0,
    umap_min_dist_viz: float = 0.1,
    umap_metric: str = 'cosine'
) -> Tuple[np.ndarray, np.ndarray, Optional[object], Dict]:
    """
    Réduit la dimensionnalité des embeddings avant clustering.
    Produit aussi une projection 2D pour visualisation.
    
    Args:
        embeddings: Matrice (n_samples, embedding_dim)
        n_components: Dimensions pour clustering (défaut: 50)
        n_components_viz: Dimensions pour visualisation (défaut: 2)
        method: 'umap' ou 'pca'
        random_state: Graine aléatoire pour reproductibilité
        umap_n_neighbors: Nombre de voisins UMAP (défaut: 15)
        umap_min_dist_cluster: min_dist pour clustering (défaut: 0.0)
        umap_min_dist_viz: min_dist pour visualisation (défaut: 0.1)
        umap_metric: Métrique de distance UMAP (défaut: 'cosine')
    
    Returns:
        (embeddings_reduced, embeddings_2d, reducer, stats)
    """
    stats = {
        'method': method,
        'original_dim': embeddings.shape[1],
        'reduced_dim': n_components,
        'viz_dim': n_components_viz
    }
    
    start_time = time.time()
    reducer = None
    
    if method == 'umap':
        try:
            import umap
            
            print(f"Réduction UMAP: {embeddings.shape[1]} → {n_components} dimensions...")
            print(f"  Paramètres: n_neighbors={umap_n_neighbors}, metric={umap_metric}")
            
            reducer = umap.UMAP(
                n_components=n_components,
                metric=umap_metric,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist_cluster,
                random_state=random_state,
                verbose=False
            )
            embeddings_reduced = reducer.fit_transform(embeddings)
            
            print(f"Projection 2D pour visualisation...")
            reducer_2d = umap.UMAP(
                n_components=n_components_viz,
                metric=umap_metric,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist_viz,
                random_state=random_state,
                verbose=False
            )
            embeddings_2d = reducer_2d.fit_transform(embeddings)
            
            stats['umap_params'] = {
                'n_components': n_components,
                'n_components_viz': n_components_viz,
                'n_neighbors': umap_n_neighbors,
                'min_dist_cluster': umap_min_dist_cluster,
                'min_dist_viz': umap_min_dist_viz,
                'metric': umap_metric,
                'random_state': random_state
            }
            
        except ImportError:
            print("UMAP non disponible, utilisation de PCA")
            method = 'pca'
    
    if method == 'pca':
        from sklearn.decomposition import PCA
        
        print(f"Réduction PCA: {embeddings.shape[1]} → {n_components} dimensions...")
        reducer = PCA(n_components=n_components, random_state=random_state)
        embeddings_reduced = reducer.fit_transform(embeddings)
        
        reducer_2d = PCA(n_components=n_components_viz, random_state=random_state)
        embeddings_2d = reducer_2d.fit_transform(embeddings)
        
        stats['pca_variance_explained'] = float(sum(reducer.explained_variance_ratio_))
        stats['pca_params'] = {'random_state': random_state}
    
    stats['reduction_time_seconds'] = round(time.time() - start_time, 2)
    print(f"  Réduction terminée en {stats['reduction_time_seconds']}s")
    
    return embeddings_reduced, embeddings_2d, reducer, stats


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.0,
    hdbscan_metric: str = 'euclidean'
) -> Tuple[np.ndarray, object, Dict]:
    """
    Applique HDBSCAN sur les embeddings.
    
    Args:
        embeddings: Matrice d'embeddings (réduite)
        min_cluster_size: Taille minimale d'un cluster
        min_samples: Nombre minimum d'échantillons dans un voisinage
        cluster_selection_epsilon: Seuil pour fusionner les clusters proches
        hdbscan_metric: Métrique de distance (défaut: 'euclidean')
                        Note: 'cosine' n'est PAS recommandé après UMAP car
                        l'espace UMAP est déjà euclidien par construction.
    
    Returns:
        (labels, clusterer, stats)
    """
    try:
        import hdbscan
    except ImportError:
        print("ERREUR: hdbscan n'est pas installé.")
        print("Installez-le avec: pip install hdbscan")
        sys.exit(1)
    
    if hdbscan_metric == 'cosine':
        print("L'espace réduit est déjà euclidien. Considérez metric='euclidean'.")
    
    print(f"Clustering HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric={hdbscan_metric})...")
    start_time = time.time()
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=hdbscan_metric,
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    labels = clusterer.fit_predict(embeddings)
    cluster_time = time.time() - start_time
    
    # Statistiques
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int((labels == -1).sum())
    
    cluster_sizes = Counter(labels)
    if -1 in cluster_sizes:
        del cluster_sizes[-1]
    
    stats = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'n_clustered': len(labels) - n_noise,
        'noise_ratio_pct': round(n_noise / len(labels) * 100, 2),
        'cluster_sizes': {int(k): int(v) for k, v in sorted(cluster_sizes.items())},
        'params': {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'cluster_selection_epsilon': cluster_selection_epsilon,
            'metric': hdbscan_metric,
            'cluster_selection_method': 'eom'
        },
        'clustering_time_seconds': round(cluster_time, 2)
    }
    
    print(f"  Clusters trouvés: {n_clusters}")
    print(f"  Segments clusterisés: {stats['n_clustered']} ({100 - stats['noise_ratio_pct']:.1f}%)")
    print(f"  Points de bruit: {n_noise} ({stats['noise_ratio_pct']}%)")
    print(f"  Temps: {cluster_time:.2f}s")
    
    return labels, clusterer, stats


def get_cluster_representatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    metadata: List[Dict],
    n_representatives: int = 15,
    probabilities: Optional[np.ndarray] = None
) -> Dict[int, List[Dict]]:
    """
    Pour chaque cluster, trouve les segments les plus représentatifs.
    
    Stratégie: utiliser les probabilités HDBSCAN si disponibles,
    sinon calcule le medoid (point réel le plus central).
    
    Args:
        embeddings: Embeddings (réduits)
        labels: Labels de cluster
        texts: Textes des segments
        metadata: Métadonnées des segments
        n_representatives: Nombre de représentants par cluster
        probabilities: Probabilités d'appartenance HDBSCAN (optionnel)
    
    Returns:
        Dict: cluster_id -> liste de représentants
    """
    print(f"Extraction des {n_representatives} segments représentatifs par cluster...")
    representatives = {}
    
    unique_clusters = sorted(set(labels) - {-1})
    
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        indices = np.where(mask)[0]
        cluster_embeddings = embeddings[mask]
        
        if probabilities is not None:
            cluster_probs = probabilities[mask]
            scores = cluster_probs  # Plus haute prob = meilleur représentant
            sorted_local_indices = np.argsort(-scores)
        else:
            n_points = len(cluster_embeddings)
            if n_points <= 200:
                dist_matrix = np.zeros((n_points, n_points))
                for i in range(n_points):
                    dist_matrix[i] = np.linalg.norm(cluster_embeddings - cluster_embeddings[i], axis=1)
                sum_distances = dist_matrix.sum(axis=1)
                sorted_local_indices = np.argsort(sum_distances)  # Plus petit = plus central
            else:
                centroid = cluster_embeddings.mean(axis=0)
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                sorted_local_indices = np.argsort(distances)
        
        n_to_take = min(n_representatives, len(indices))
        top_indices = sorted_local_indices[:n_to_take]
        
        representatives[cluster_id] = []
        for local_idx in top_indices:
            global_idx = indices[local_idx]
            rep_info = {
                'global_idx': int(global_idx),
                'text': texts[global_idx],
                'doc_id': metadata[global_idx].get('doc_id'),
                'period': metadata[global_idx].get('period'),
                'year': metadata[global_idx].get('year')
            }
            if probabilities is not None:
                rep_info['probability'] = float(probabilities[global_idx])
            representatives[cluster_id].append(rep_info)
    
    return representatives


def analyze_clusters_by_period(
    labels: np.ndarray,
    metadata: List[Dict]
) -> Dict:
    """
    Analyse la distribution des clusters par période.
    Permet de voir quels narratifs dominent à quelle époque.
    
    Returns:
        Dict avec distribution, pourcentages normalisés, clusters dominants
    """
    print("Analyse de la distribution par période...")
    
    distribution = defaultdict(lambda: defaultdict(int))
    period_totals = defaultdict(int)
    
    for label, meta in zip(labels, metadata):
        period = meta.get('period', 'unknown')
        period_totals[period] += 1
        if label != -1:
            distribution[period][int(label)] += 1
    
    normalized = {}
    for period in sorted(distribution.keys()):
        total_clustered = sum(distribution[period].values())
        if total_clustered > 0:
            normalized[period] = {
                int(cluster): round(count / total_clustered * 100, 2)
                for cluster, count in sorted(distribution[period].items())
            }
        else:
            normalized[period] = {}
    
    dominant_by_period = {}
    for period, clusters in normalized.items():
        if clusters:
            dominant = max(clusters.items(), key=lambda x: x[1])
            dominant_by_period[period] = {
                'cluster': dominant[0], 
                'percentage': dominant[1]
            }
    
    cluster_by_period = defaultdict(lambda: defaultdict(int))
    for label, meta in zip(labels, metadata):
        if label != -1:
            cluster_by_period[int(label)][meta.get('period', 'unknown')] += 1
    
    return {
        'distribution': {k: dict(v) for k, v in distribution.items()},
        'normalized_pct': normalized,
        'dominant_by_period': dominant_by_period,
        'cluster_by_period': {k: dict(v) for k, v in cluster_by_period.items()},
        'period_totals': dict(period_totals)
    }


def save_results(
    output_dir: Path,
    labels: np.ndarray,
    embeddings_2d: np.ndarray,
    metadata: List[Dict],
    texts: List[str],
    clustering_stats: Dict,
    reduction_stats: Dict,
    representatives: Dict,
    period_analysis: Dict,
    total_time: float,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, Path]:
    """
    Sauvegarde tous les résultats du clustering.
    
    Returns:
        Dict des chemins créés
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    
    print(f"\nSauvegarde des résultats dans {output_dir}...")
    
    labels_path = output_dir / "cluster_labels.npy"
    np.save(labels_path, labels)
    paths['labels'] = labels_path
    print(f"  Labels: {labels_path}")
    
    # Projection 2D pour visualisation
    viz_path = output_dir / "embeddings_2d.npy"
    np.save(viz_path, embeddings_2d)
    paths['viz'] = viz_path
    print(f"  Projection 2D: {viz_path}")
    
    # Probabilities numpy
    if probabilities is not None:
        prob_path = output_dir / "cluster_probabilities.npy"
        np.save(prob_path, probabilities)
        paths['probabilities'] = prob_path
        print(f"  Probabilities: {prob_path}")
    
    # Statistiques complètes
    full_stats = {
        'clustering': clustering_stats,
        'reduction': reduction_stats,
        'total_time_seconds': round(total_time, 2)
    }
    stats_path = output_dir / "clustering_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(full_stats, f, ensure_ascii=False, indent=2)
    paths['stats'] = stats_path
    print(f"  Statistiques: {stats_path}")
    
    # Représentants des clusters
    repr_path = output_dir / "cluster_representatives.json"
    with open(repr_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in representatives.items()}, f, ensure_ascii=False, indent=2)
    paths['representatives'] = repr_path
    print(f"  Représentants: {repr_path}")
    
    # Analyse par période
    period_path = output_dir / "period_analysis.json"
    with open(period_path, 'w', encoding='utf-8') as f:
        json.dump(period_analysis, f, ensure_ascii=False, indent=2)
    paths['period'] = period_path
    print(f"  Analyse périodes: {period_path}")
    
    csv_path = output_dir / "segments_with_clusters.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['idx', 'cluster', 'is_noise', 'probability', 'doc_id', 'period', 'year', 'x_2d', 'y_2d', 'text_preview']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, (label, meta, text) in enumerate(zip(labels, metadata, texts)):
            is_noise = (label == -1)
            writer.writerow({
                'idx': i,
                'cluster': int(label),
                'is_noise': is_noise,
                'probability': round(float(probabilities[i]), 4) if probabilities is not None else '',
                'doc_id': meta.get('doc_id', ''),
                'period': meta.get('period', ''),
                'year': meta.get('year', ''),
                'x_2d': round(embeddings_2d[i, 0], 4),
                'y_2d': round(embeddings_2d[i, 1], 4),
                'text_preview': text[:150].replace('\n', ' ')
            })
    paths['csv'] = csv_path
    print(f"  CSV complet: {csv_path}")
    
    if probabilities is not None:
        clustered_mask = labels != -1
        if clustered_mask.sum() > 0:
            clustered_probs = probabilities[clustered_mask]
            print(f"\n  Probability stats :")
            print(f"    Mean: {clustered_probs.mean():.3f}")
            print(f"    Median: {np.median(clustered_probs):.3f}")
            print(f"    Min: {clustered_probs.min():.3f}")
            print(f"    Max: {clustered_probs.max():.3f}")
            print(f"    Points à forte confiance (> 0.8) : {(clustered_probs > 0.8).sum()} segments")
            print(f"    Points à faible confiance (< 0.5) : {(clustered_probs < 0.5).sum()} segments")
    
    return paths


def process_clustering(
    embeddings_dir: Path,
    output_dir: Path,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    reduce_dim: bool = True,
    n_components: int = 50,
    n_representatives: int = 15,
    umap_n_neighbors: int = 15,
    umap_min_dist_cluster: float = 0.0,
    umap_min_dist_viz: float = 0.1,
    umap_metric: str = 'cosine',
    hdbscan_metric: str = 'euclidean'
) -> Dict:
    """
    Pipeline complet de clustering.
    
    Args:
        embeddings_dir: Répertoire des embeddings (contient all_vectors.npy, etc.)
        output_dir: Répertoire de sortie
        min_cluster_size: Taille minimale d'un cluster HDBSCAN
        min_samples: Paramètre min_samples HDBSCAN
        reduce_dim: Appliquer réduction de dimension
        n_components: Dimensions pour clustering
        n_representatives: Nombre de représentants par cluster
        umap_n_neighbors: Voisins UMAP
        umap_min_dist_cluster: min_dist pour clustering
        umap_min_dist_viz: min_dist pour visualisation
        umap_metric: Métrique UMAP
        hdbscan_metric: Métrique HDBSCAN
    
    Returns:
        Statistiques du clustering
    """
    total_start = time.time()
    
    embeddings, metadata, texts = load_embeddings(embeddings_dir, prefix="all")
    
    # Réduction de dimension
    if reduce_dim:
        embeddings_reduced, embeddings_2d, reducer, reduction_stats = reduce_dimensions(
            embeddings, 
            n_components=n_components,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist_cluster=umap_min_dist_cluster,
            umap_min_dist_viz=umap_min_dist_viz,
            umap_metric=umap_metric
        )
    else:
        # Fallback: PCA pour la projection 2D
        from sklearn.decomposition import PCA
        embeddings_reduced = embeddings
        pca_2d = PCA(n_components=2, random_state=42)
        embeddings_2d = pca_2d.fit_transform(embeddings)
        reduction_stats = {'method': 'none', 'viz_method': 'pca_2d'}
    
    # Clustering HDBSCAN
    labels, clusterer, clustering_stats = cluster_hdbscan(
        embeddings_reduced,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        hdbscan_metric=hdbscan_metric
    )
    
    clustering_stats['input_space'] = 'reduced' if reduce_dim else 'original'
    clustering_stats['input_space_method'] = reduction_stats.get('method', 'none') if reduce_dim else 'none'
    
    probabilities = clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None
    
    representatives = get_cluster_representatives(
        embeddings_reduced, labels, texts, metadata, n_representatives,
        probabilities=probabilities
    )
    
    # Analyse par période
    period_analysis = analyze_clusters_by_period(labels, metadata)
    
    # Sauvegarder (avec les probabilities)
    total_time = time.time() - total_start
    
    save_results(
        output_dir=output_dir,
        labels=labels,
        embeddings_2d=embeddings_2d,
        metadata=metadata,
        texts=texts,
        clustering_stats=clustering_stats,
        reduction_stats=reduction_stats,
        representatives=representatives,
        period_analysis=period_analysis,
        total_time=total_time,
        probabilities=probabilities
    )
    
    return {
        'clustering': clustering_stats,
        'reduction': reduction_stats,
        'period_analysis': period_analysis,
        'total_time_seconds': round(total_time, 2)
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Étape 6: Clusterisation HDBSCAN des embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python step6_cluster.py ./embeddings_out ./clustering_out
  python step6_cluster.py ./embeddings_out ./clustering_out --min-cluster-size 15 --min-samples 8
  python step6_cluster.py ./embeddings_out ./clustering_out --umap-neighbors 20 --umap-metric euclidean
        """
    )
    
    parser.add_argument('embeddings_dir', type=Path,
                        help='Répertoire contenant all_vectors.npy, all_metadata.json, all_texts.json')
    parser.add_argument('output_dir', type=Path,
                        help='Répertoire de sortie pour les résultats')
    
    # HDBSCAN paramètres
    parser.add_argument('--min-cluster-size', type=int, default=10,
                        help='Taille minimale d\'un cluster (défaut: 10)')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='Densité minimale HDBSCAN (défaut: 5)')
    parser.add_argument('--hdbscan-metric', type=str, default='euclidean',
                        choices=['euclidean', 'manhattan'],
                        help='Métrique HDBSCAN (défaut: euclidean). '
                             'Note: cosine non recommandé après réduction UMAP/PCA')
    
    # UMAP paramètres
    parser.add_argument('--umap-neighbors', type=int, default=15,
                        help='Nombre de voisins UMAP (défaut: 15)')
    parser.add_argument('--umap-min-dist', type=float, default=0.0,
                        help='min_dist UMAP pour clustering (défaut: 0.0)')
    parser.add_argument('--umap-min-dist-viz', type=float, default=0.1,
                        help='min_dist UMAP pour visualisation (défaut: 0.1)')
    parser.add_argument('--umap-metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean', 'manhattan', 'correlation'],
                        help='Métrique UMAP (défaut: cosine)')
    
    # Autres paramètres
    parser.add_argument('--n-components', type=int, default=50,
                        help='Dimensions après réduction UMAP (défaut: 50)')
    parser.add_argument('--n-representatives', type=int, default=15,
                        help='Nombre de représentants par cluster (défaut: 15)')
    parser.add_argument('--no-reduce', action='store_true',
                        help='Ne pas appliquer la réduction de dimension')
    
    args = parser.parse_args()
    
    if not args.embeddings_dir.exists():
        print(f"ERREUR: Répertoire non trouvé: {args.embeddings_dir}")
        sys.exit(1)
    
    results = process_clustering(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        reduce_dim=not args.no_reduce,
        n_components=args.n_components,
        n_representatives=args.n_representatives,
        umap_n_neighbors=args.umap_neighbors,
        umap_min_dist_cluster=args.umap_min_dist,
        umap_min_dist_viz=args.umap_min_dist_viz,
        umap_metric=args.umap_metric,
        hdbscan_metric=args.hdbscan_metric
    )
    
    print(f"\n{'='*60}")
    print("CLUSTERING TERMINÉ")
    print(f"{'='*60}")
    print(f"Clusters trouvés: {results['clustering']['n_clusters']}")
    print(f"Segments clusterisés: {results['clustering']['n_clustered']}")
    print(f"Bruit: {results['clustering']['n_noise']} ({results['clustering']['noise_ratio_pct']}%)")
    print(f"Temps total: {results['total_time_seconds']}s")
    
    print(f"\nDistribution par période (clusters dominants):")
    for period, info in sorted(results['period_analysis']['dominant_by_period'].items()):
        print(f"  {period}: Cluster {info['cluster']} ({info['percentage']}%)")
    
    print(f"\nTailles des clusters:")
    for cluster_id, size in sorted(results['clustering']['cluster_sizes'].items()):
        print(f"  Cluster {cluster_id}: {size} segments")


if __name__ == '__main__':
    main()