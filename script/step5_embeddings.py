"""
Étape 5 : Génération des embeddings avec SBERT
Utilise un modèle sentence-transformers optimisé pour le russe.

Modèle recommandé:
- ai-forever/sbert_large_nlu_ru (768 dim, meilleur pour le russe)
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

DEFAULT_MODEL = "ai-forever/sbert_large_nlu_ru"


def load_model(model_name: str = DEFAULT_MODEL):
    """
    Charge le modèle SBERT.
    
    Args:
        model_name: Nom du modèle HuggingFace
    
    Returns:
        SentenceTransformer model
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERREUR: sentence-transformers n'est pas installé.")
        print("Installez-le avec: pip install sentence-transformers")
        sys.exit(1)
    
    print(f"Chargement du modèle {model_name}...")
    
    start_time = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - start_time
    
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Modèle chargé en {load_time:.1f}s")
    print(f"Dimension des embeddings: {embedding_dim}")
    
    return model, {
        'model_name': model_name,
        'embedding_dim': embedding_dim,
        'load_time_seconds': round(load_time, 2)
    }


def encode_segments(
    segments: List[Dict], 
    model,
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Encode tous les segments en embeddings.
    
    Args:
        segments: Liste de segments (dicts avec clé 'text')
        model: Modèle SentenceTransformer
        batch_size: Taille des batches pour l'encodage
        normalize: Normaliser les vecteurs (L2) pour cosine similarity
        show_progress: Afficher une barre de progression
    
    Returns:
        Tuple (embeddings array, stats dict)
    """
    texts = [seg['text'] for seg in segments]
    
    print(f"Encodage de {len(texts)} segments (batch_size={batch_size})...")
    start_time = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize  # Normalisation L2 pour cosine similarity
    )
    
    encode_time = time.time() - start_time
    speed = len(texts) / encode_time
    print(f"Encodage terminé en {encode_time:.1f}s ({speed:.1f} segments/s)")
    
    return embeddings, {
        'encode_time_seconds': round(encode_time, 2),
        'segments_per_second': round(speed, 2),
        'normalized': normalize
    }


def group_by_period(segments: List[Dict]) -> Dict[str, List[Tuple[int, Dict]]]:
    """
    Groupe les segments par période avec leurs indices originaux.
    
    Returns:
        Dict: period -> [(original_idx, segment), ...]
    """
    from collections import defaultdict
    
    by_period = defaultdict(list)
    for i, seg in enumerate(segments):
        by_period[seg['period']].append((i, seg))
    
    return dict(by_period)


def save_embeddings(
    embeddings: np.ndarray, 
    segments: List[Dict],
    output_dir: Path,
    prefix: str = "embeddings"
) -> Dict[str, Path]:
    """
    Sauvegarde les embeddings et les métadonnées associées.
    
    Fichiers créés:
    - {prefix}_vectors.npy: Matrice numpy des embeddings
    - {prefix}_metadata.json: Métadonnées des segments
    - {prefix}_texts.json: Textes complets
    - {prefix}_index.csv: Index pour liaison avec clustering
    
    Returns:
        Dict des chemins créés
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    
    # Sauvegarder les vecteurs
    vectors_path = output_dir / f"{prefix}_vectors.npy"
    np.save(vectors_path, embeddings)
    paths['vectors'] = vectors_path
    print(f"  Vecteurs: {vectors_path}")
    
    # Sauvegarder les métadonnées
    metadata = []
    for i, seg in enumerate(segments):
        meta = {
            'idx': i,
            'doc_id': seg['doc_id'],
            'period': seg['period'],
            'year': seg.get('year'),
            'level': seg.get('level'),
            'country': seg.get('country'),
            'paragraph_idx': seg['paragraph_idx'],
            'segment_idx': seg['segment_idx'],
            'char_count': seg['char_count'],
            'word_count': seg.get('word_count', len(seg['text'].split())),
            'sentence_count': seg.get('sentence_count'),
            'text_preview': seg['text'][:200] + '...' if len(seg['text']) > 200 else seg['text']
        }
        for key in ['filter_score', 'filter_passed_reason']:
            if key in seg:
                meta[key] = seg[key]
        metadata.append(meta)
    
    metadata_path = output_dir / f"{prefix}_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    paths['metadata'] = metadata_path
    print(f"  Métadonnées: {metadata_path}")
    
    texts_path = output_dir / f"{prefix}_texts.json"
    texts = [{'idx': i, 'text': seg['text']} for i, seg in enumerate(segments)]
    with open(texts_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    paths['texts'] = texts_path
    print(f"  Textes: {texts_path}")
    
    index_path = output_dir / f"{prefix}_index.csv"
    with open(index_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['idx', 'doc_id', 'period', 'year', 'paragraph_idx', 'segment_idx', 
                      'char_count', 'word_count', 'filter_score']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(metadata)
    paths['index'] = index_path
    print(f"  Index CSV: {index_path}")
    
    return paths


def save_embeddings_by_period(
    all_embeddings: np.ndarray,
    segments: List[Dict],
    output_dir: Path
) -> Dict[str, int]:
    """
    Sauvegarde les embeddings séparément pour chaque période.
    
    Returns:
        Dict: period -> nombre de segments
    """
    by_period = group_by_period(segments)
    counts = {}
    
    print("\nSauvegarde par période:")
    for period in sorted(by_period.keys()):
        items = by_period[period]
        indices = [idx for idx, _ in items]
        period_segments = [seg for _, seg in items]
        period_embeddings = all_embeddings[indices]
        
        period_dir = output_dir / "by_period" / period
        save_embeddings(period_embeddings, period_segments, period_dir, prefix="embeddings")
        counts[period] = len(items)
    
    return counts


def process_embeddings(
    input_path: Path, 
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    normalize: bool = True,
    save_by_period: bool = True
) -> Dict:
    """
    Pipeline complet: charge, encode, sauvegarde.
    
    Args:
        input_path: Chemin vers le JSON des segments
        output_dir: Répertoire de sortie
        model_name: Nom du modèle à utiliser
        batch_size: Taille des batches
        normalize: Normaliser les embeddings (recommandé pour cosine similarity)
        save_by_period: Sauvegarder aussi par période
    
    Returns:
        Statistiques du processus
    """
    total_start = time.time()
    
    print(f"Chargement des segments depuis {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    print(f"  {len(segments)} segments chargés")
    
    model, model_stats = load_model(model_name)
    
    embeddings, encode_stats = encode_segments(
        segments, model, batch_size, normalize=normalize
    )
    
    print("\nSauvegarde globale:")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_embeddings(embeddings, segments, output_dir, prefix="all")
    
    period_counts = {}
    if save_by_period:
        period_counts = save_embeddings_by_period(embeddings, segments, output_dir)
    
    total_time = time.time() - total_start
    
    stats = {
        'input_file': str(input_path),
        'output_dir': str(output_dir),
        'total_segments': len(segments),
        'embedding_dim': embeddings.shape[1],
        'embeddings_shape': list(embeddings.shape),
        'normalized': normalize,
        'model': model_stats,
        'encoding': encode_stats,
        'by_period': period_counts,
        'total_time_seconds': round(total_time, 2)
    }
    
    stats_path = output_dir / "embeddings_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nStatistiques: {stats_path}")
    
    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 step5_embeddings.py <input_segments.json> [output_dir] [model_name]")
        print(f"\nExemple: python step5_embeddings.py ./segments_final.json")
        print(f"\nSortie par défaut: ./embeddings_out/")
        print(f"Modèle par défaut: {DEFAULT_MODEL}")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("embeddings_out")
    model_name = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL
    
    if not input_path.exists():
        print(f"ERREUR: Fichier non trouvé: {input_path}")
        sys.exit(1)
    
    stats = process_embeddings(input_path, output_dir, model_name)
    
    print(f"\n{'='*60}")
    print("ENCODAGE TERMINÉ")
    print(f"{'='*60}")
    print(f"Segments encodés: {stats['total_segments']}")
    print(f"Dimension: {stats['embedding_dim']}")
    print(f"Normalisé: {stats['normalized']}")
    print(f"Modèle: {stats['model']['model_name']}")
    print(f"Temps total: {stats['total_time_seconds']}s")
    
    if stats['by_period']:
        print(f"\nPar période:")
        for period, count in sorted(stats['by_period'].items()):
            print(f"  {period}: {count} embeddings")


if __name__ == '__main__':
    main()
