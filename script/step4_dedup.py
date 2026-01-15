"""
Étape 4 : Déduplication des segments
Supprime les segments identiques ou quasi-identiques.

Méthodes:
1. Déduplication exacte (hash MD5)
2. Déduplication floue (similarité de Jaccard sur n-grammes)
"""

import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict


NGRAM_SIZE = 3  # Taille des n-grammes pour la déduplication floue
MIN_LENGTH_FOR_FUZZY = 10  # Longueur minimale normalisée pour fuzzy matching


@dataclass
class DuplicateInfo:
    """Information sur un doublon détecté."""
    removed_text: str           # Texte du segment supprimé (tronqué, original)
    kept_text: str              # Texte du segment conservé (tronqué, original)
    removed_doc_id: str         # ID du document supprimé
    kept_doc_id: str            # ID du document conservé
    similarity: float           # Score de similarité (1.0 pour exact)
    method: str                 # 'exact' ou 'fuzzy'
    removed_normalized: str = ""  # Texte normalisé du segment supprimé
    kept_normalized: str = ""     # Texte normalisé du segment conservé (pour validation manuelle)


def normalize_for_comparison(text: str) -> str:
    """
    Normalise le texte pour la comparaison.
    Supprime la ponctuation, met en minuscules, unifie les espaces.
    """
    text = text.lower()
    # Normaliser ё → е pour le russe
    text = text.replace('ё', 'е')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_text_hash(text: str) -> str:
    """Retourne le hash MD5 du texte normalisé."""
    normalized = normalize_for_comparison(text)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def get_ngrams(text: str, n: int = NGRAM_SIZE) -> Set[str]:
    """Extrait les n-grammes de caractères du texte (normalise d'abord)."""
    normalized = normalize_for_comparison(text)
    return get_ngrams_from_normalized(normalized, n)


def get_ngrams_from_normalized(normalized: str, n: int = NGRAM_SIZE) -> Set[str]:
    """
    Extrait les n-grammes d'un texte normalisé (évite double normalisation).
    
    Note: Le cas len(normalized) < n retourne {normalized}, mais en pratique
    il est filtré par MIN_LENGTH_FOR_FUZZY dans deduplicate_fuzzy().
    Cette condition reste pour la compatibilité avec d'autres usages de get_ngrams().
    """
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[i:i+n] for i in range(len(normalized) - n + 1)}


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calcule la similarité de Jaccard entre deux ensembles."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def truncate_text(text: str, max_len: int = 150) -> str:
    """Tronque le texte pour le logging."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + '...'


def get_segment_text(segment: Dict) -> Optional[str]:
    """
    Extrait le texte d'un segment de manière sûre.
    Retourne None si le segment est invalide (pas de 'text' ou type incorrect).
    """
    text = segment.get('text')
    if text is None or not isinstance(text, str):
        return None
    return text


def deduplicate_exact(segments: List[Dict], 
                      log_duplicates: bool = True) -> Tuple[List[Dict], int, List[DuplicateInfo]]:
    """
    Supprime les doublons exacts basés sur le hash du texte.
    Garde le premier segment rencontré pour chaque hash.
    
    Returns:
        (unique_segments, duplicates_removed, duplicate_log)
    """
    seen_hashes: Dict[str, Dict] = {}
    unique_segments = []
    duplicates_removed = 0
    duplicate_log: List[DuplicateInfo] = []
    skipped_invalid = 0
    
    for segment in segments:
        text = get_segment_text(segment)
        if text is None:
            skipped_invalid += 1
            continue
            
        text_hash = get_text_hash(text)
        
        if text_hash not in seen_hashes:
            seen_hashes[text_hash] = segment
            unique_segments.append(segment)
        else:
            duplicates_removed += 1
            if log_duplicates:
                kept = seen_hashes[text_hash]
                normalized = normalize_for_comparison(text)
                kept_text = get_segment_text(kept) or ""
                kept_normalized = normalize_for_comparison(kept_text)
                
                duplicate_log.append(DuplicateInfo(
                    removed_text=truncate_text(text),
                    kept_text=truncate_text(kept_text),
                    removed_doc_id=segment.get('doc_id', 'unknown'),
                    kept_doc_id=kept.get('doc_id', 'unknown'),
                    similarity=1.0,
                    method='exact',
                    removed_normalized=truncate_text(normalized, 100),
                    kept_normalized=truncate_text(kept_normalized, 100)
                ))
    
    if skipped_invalid > 0:
        print(f"[exact] {skipped_invalid} segments ignorés (pas de champ 'text' valide)")
    
    return unique_segments, duplicates_removed, duplicate_log


def deduplicate_fuzzy(segments: List[Dict], 
                      threshold: float = 0.85,
                      log_duplicates: bool = True,
                      length_tolerance: float = 0.5,
                      min_length: int = MIN_LENGTH_FOR_FUZZY,
                      ngram_size: int = NGRAM_SIZE) -> Tuple[List[Dict], int, List[DuplicateInfo]]:
    """
    Supprime les quasi-doublons basés sur la similarité de Jaccard.
    
    Algorithme:
    1. Pré-calcul des n-grammes et longueurs
    2. Filtrage par longueur avant calcul de Jaccard (optimisation)
    3. Pour chaque segment, recherche du meilleur match parmi tous les candidats
    
    Args:
        segments: Liste des segments à dédupliquer
        threshold: Seuil de similarité (défaut: 0.85)
        log_duplicates: Si True, enregistre les paires de doublons
        length_tolerance: Ratio min/max de longueur minimum pour comparer (défaut: 0.5)
        min_length: Longueur minimale normalisée pour fuzzy (défaut: MIN_LENGTH_FOR_FUZZY)
        ngram_size: Taille des n-grammes (défaut: NGRAM_SIZE)
    
    Notes:
        - Complexité O(n²) dans le pire cas
        - Textes < min_length: exclus du fuzzy (peuvent être supprimés par exact)
        - Pas d'early exit: on cherche vraiment le MEILLEUR match (similarité maximale)
    """
    unique_segments: List[Dict] = []
    unique_ngrams: List[Set[str]] = []
    unique_lengths: List[int] = []
    duplicates_removed = 0
    duplicate_log: List[DuplicateInfo] = []
    
    # Stats d'optimisation
    comparisons_total = 0
    comparisons_skipped_length = 0
    skipped_too_short = 0
    skipped_empty = 0
    skipped_invalid = 0
    
    for segment in segments:
        text = get_segment_text(segment)
        if text is None:
            skipped_invalid += 1
            continue
        
        normalized = normalize_for_comparison(text)
        text_len = len(normalized)
        
        if not normalized:
            unique_segments.append(segment)
            unique_ngrams.append(set())
            unique_lengths.append(0)
            skipped_empty += 1
            continue
        
        if text_len < min_length:
            unique_segments.append(segment)
            unique_ngrams.append(set())
            unique_lengths.append(text_len)
            skipped_too_short += 1
            continue
        
        ngrams = get_ngrams_from_normalized(normalized, n=ngram_size)
        
        is_duplicate = False
        best_match_idx = -1
        best_similarity = 0.0
        
        for idx, (existing_ngrams, existing_len) in enumerate(zip(unique_ngrams, unique_lengths)):
            if existing_len < min_length:
                continue
                
            comparisons_total += 1
            
            # Jaccard(A,B) <= |A|/|B| si |A| <= |B|
            # Donc si len_ratio < threshold, Jaccard ne peut jamais atteindre threshold
            len_ratio = min(text_len, existing_len) / max(text_len, existing_len) if max(text_len, existing_len) > 0 else 0
            effective_len_threshold = max(length_tolerance, threshold)
            if len_ratio < effective_len_threshold:
                comparisons_skipped_length += 1
                continue
            
            similarity = jaccard_similarity(ngrams, existing_ngrams)
            if similarity >= threshold:
                is_duplicate = True
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = idx
        
        if is_duplicate:
            duplicates_removed += 1
            if log_duplicates and best_match_idx >= 0:
                kept = unique_segments[best_match_idx]
                kept_text = get_segment_text(kept) or ""
                kept_normalized = normalize_for_comparison(kept_text)
                
                duplicate_log.append(DuplicateInfo(
                    removed_text=truncate_text(text),
                    kept_text=truncate_text(kept_text),
                    removed_doc_id=segment.get('doc_id', 'unknown'),
                    kept_doc_id=kept.get('doc_id', 'unknown'),
                    similarity=round(best_similarity, 3),
                    method='fuzzy',
                    removed_normalized=truncate_text(normalized, 100),
                    kept_normalized=truncate_text(kept_normalized, 100)
                ))
        else:
            unique_segments.append(segment)
            unique_ngrams.append(ngrams)
            unique_lengths.append(text_len)
    
    if comparisons_total > 0:
        skip_rate = comparisons_skipped_length / comparisons_total * 100
        print(f"  [fuzzy] Comparaisons: {comparisons_total}, skip (longueur): {comparisons_skipped_length} ({skip_rate:.1f}%)")
    if skipped_empty > 0 or skipped_too_short > 0:
        print(f"  [fuzzy] Ignorés: {skipped_empty} vides, {skipped_too_short} trop courts (<{min_length} chars)")
    
    if skipped_invalid > 0:
        print(f"  [fuzzy] {skipped_invalid} segments ignorés (pas de champ 'text' valide)")
    
    return unique_segments, duplicates_removed, duplicate_log


def deduplicate_by_period(segments: List[Dict], 
                          exact: bool = True, 
                          fuzzy: bool = True,
                          fuzzy_threshold: float = 0.85,
                          log_duplicates: bool = True) -> Tuple[List[Dict], Dict, List[Dict]]:
    """
    Applique la déduplication par période.
    Les doublons sont cherchés au sein de chaque période séparément.
    
    Cela permet de garder des formulations similaires entre périodes
    (qui peuvent être intéressantes pour l'analyse diachronique).
    
    Returns:
        (deduplicated_segments, stats, duplicate_log)
    """
    # Grouper par période
    by_period = defaultdict(list)
    missing_period_count = 0
    for segment in segments:
        period = segment.get('period', 'unknown')
        if period == 'unknown' and 'period' not in segment:
            missing_period_count += 1
        by_period[period].append(segment)
    
    if missing_period_count > 0:
        print(f"  {missing_period_count} segments sans champ 'period' (groupés dans 'unknown')")
    
    deduplicated = []
    all_duplicates: List[Dict] = []
    
    stats = {
        'by_period': {},
        'exact_removed': 0,
        'fuzzy_removed': 0,
        'total_removed': 0
    }
    
    for period in sorted(by_period.keys()):
        period_segments = by_period[period]
        original_count = len(period_segments)
        exact_removed = 0
        fuzzy_removed = 0
        period_duplicates: List[DuplicateInfo] = []
        
        print(f"\n  Période '{period}' ({original_count} segments):")
        
        # Déduplication exacte
        if exact:
            period_segments, exact_removed, exact_dups = deduplicate_exact(
                period_segments, log_duplicates=log_duplicates
            )
            period_duplicates.extend(exact_dups)
        
        # Déduplication floue
        if fuzzy:
            period_segments, fuzzy_removed, fuzzy_dups = deduplicate_fuzzy(
                period_segments, threshold=fuzzy_threshold, log_duplicates=log_duplicates
            )
            period_duplicates.extend(fuzzy_dups)
        
        deduplicated.extend(period_segments)
        
        # Ajouter la période aux logs de doublons
        for dup in period_duplicates:
            dup_dict = asdict(dup)
            dup_dict['period'] = period
            all_duplicates.append(dup_dict)
        
        stats['by_period'][period] = {
            'original': original_count,
            'after_dedup': len(period_segments),
            'exact_removed': exact_removed,
            'fuzzy_removed': fuzzy_removed
        }
        stats['exact_removed'] += exact_removed
        stats['fuzzy_removed'] += fuzzy_removed
    
    stats['total_removed'] = stats['exact_removed'] + stats['fuzzy_removed']
    stats['input_count'] = len(segments)
    stats['output_count'] = len(deduplicated)
    stats['dedup_rate'] = round(stats['total_removed'] / stats['input_count'] * 100, 2) if stats['input_count'] > 0 else 0
    
    return deduplicated, stats, all_duplicates


def process_deduplication(input_path: Path, output_path: Path,
                          exact: bool = True, fuzzy: bool = True,
                          fuzzy_threshold: float = 0.85,
                          log_duplicates: bool = True) -> Dict:
    """
    Charge, déduplique et sauvegarde les segments.
    
    Génère également:
    - deduplication_stats.json : statistiques détaillées
    - deduplication_duplicates.json : log des paires de doublons (pour validation)
    """
    print(f"Chargement des segments depuis {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    print(f"\nDéduplication (exact={exact}, fuzzy={fuzzy}, threshold={fuzzy_threshold})...")
    print(f"Configuration: NGRAM_SIZE={NGRAM_SIZE}, MIN_LENGTH_FOR_FUZZY={MIN_LENGTH_FOR_FUZZY}")
    
    deduplicated, stats, duplicate_log = deduplicate_by_period(
        segments, exact=exact, fuzzy=fuzzy, 
        fuzzy_threshold=fuzzy_threshold, log_duplicates=log_duplicates
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(deduplicated, f, ensure_ascii=False, indent=2)
    print(f"\nSegments dédupliqués exportés: {output_path}")
    
    stats_path = output_path.parent / "deduplication_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistiques exportées: {stats_path}")
    
    if log_duplicates and duplicate_log:
        duplicates_path = output_path.parent / "deduplication_duplicates.json"
        
        exact_all = [d for d in duplicate_log if d['method'] == 'exact']
        fuzzy_all = [d for d in duplicate_log if d['method'] == 'fuzzy']
        fuzzy_low_conf = [d for d in fuzzy_all if d['similarity'] < 0.90]
        
        duplicates_report = {
            'description': 'Paires de doublons supprimés (pour validation)',
            'summary': {
                'exact_pairs_total': len(exact_all),
                'fuzzy_pairs_total': len(fuzzy_all),
                'fuzzy_low_confidence_total': len(fuzzy_low_conf),
            },
            'by_method_sample': {
                'exact_first_100': exact_all[:100],
                'fuzzy_first_100': fuzzy_all[:100],
            },
            'fuzzy_low_confidence_sample': fuzzy_low_conf[:50],
        }
        
        with open(duplicates_path, 'w', encoding='utf-8') as f:
            json.dump(duplicates_report, f, ensure_ascii=False, indent=2)
        print(f"Log des doublons exporté: {duplicates_path}")
        
        if duplicates_report['summary']['fuzzy_low_confidence_total'] > 0:
            print(f"  {duplicates_report['summary']['fuzzy_low_confidence_total']} paires fuzzy avec similarité < 0.90")
    
    return stats


if __name__ == '__main__':
    import sys
    
    args = sys.argv[1:]
    no_log = '--no-log' in args
    if no_log:
        args.remove('--no-log')
    
    if len(args) < 2:
        print("Usage: python step4_deduplicate.py <input.json> <output.json> [fuzzy_threshold] [--no-log]")
        print("")
        print("Exemple: python step4_deduplicate.py ./segments_stalin.json ./segments_final.json 0.85")
        print("         python step4_deduplicate.py ./segments.json ./deduped.json 0.90 --no-log")
        print("")
        print("Options:")
        print("  --no-log  Ne pas générer le fichier de log des doublons")
        sys.exit(1)
    
    input_path = Path(args[0])
    output_path = Path(args[1])
    threshold = float(args[2]) if len(args) > 2 else 0.85
    
    stats = process_deduplication(
        input_path, output_path, 
        fuzzy_threshold=threshold,
        log_duplicates=not no_log
    )
    
    print(f"\n{'='*50}")
    print(f"Déduplication terminée")
    print(f"  Segments en entrée: {stats['input_count']}")
    print(f"  Segments en sortie: {stats['output_count']}")
    print(f"  Taux de déduplication: {stats['dedup_rate']}%")
    print(f"  Doublons exacts supprimés: {stats['exact_removed']}")
    print(f"  Quasi-doublons supprimés: {stats['fuzzy_removed']}")
    print(f"\nPar période:")
    for period, pstats in sorted(stats['by_period'].items()):
        print(f"  {period}: {pstats['original']} → {pstats['after_dedup']} "
              f"(-{pstats['exact_removed']} exact, -{pstats['fuzzy_removed']} fuzzy)")