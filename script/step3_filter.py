"""
Étape 3 : Filtrage thématique du corpus
Seuls les segments liés à Staline/période stalinienne sont retenus.

Critères de filtrage:
- Marqueurs lexicaux explicites (Сталин, ГУЛАГ, репрессии etc.)
- Dates de la période stalinienne 1924-1953 (uniquement en présence d'autres marqueurs)
- Événements clés de l'époque
"""

import re
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

STALIN_MARKERS = [
    r'\bсталин(?!град)\w*',
    r'\bиосиф\s+виссарионович',
    r'\bджугашвили',
]

# Marqueurs faibles de Staline (poids réduit: +1)
WEAK_STALIN_MARKERS = [
    r'\bкоба\b',
]

# Culte de la personnalité
CULT_MARKERS = [
    r'культ\s+личности',
    r'вождь\s+народ(?:а|ов)?',
    r'великий\s+вождь',
    r'отец\s+народ(?:а|ов)?',
    r'гениальн\w*\s+вождь',
    r'мудр\w*\s+(?:руководств|вождь|сталин)',
]

# Répressions
REPRESSION_MARKERS = [
    r'\bрепресс\w*',
    r'\bгулаг\w*',
    r'\bнквд\b',
    r'\bогпу\b',
    r'\bнкгб\b',
    r'\bбольшой\s+террор',
    r'\bмассов\w+\s+террор',
    r'\bполитическ\w+\s+террор',
    r'\bчистк[аи]\b',
    r'\bраскулачива\w*',
    r'\bвраг\w*\s+народа',
    r'\bтройк[аи]\b',
    r'\bлагер[яьей]\w*',
    r'\bссылк\w*',
    r'\bдепортац\w*',
    r'\bвыселен\w*',
    r'\bрасстрел\w*',
    r'\bарест\w*(?:\s+\w+){0,2}\s*(?:1930|1937|1938|194\d)',
]

# Événements clés
EVENT_MARKERS = [
    r'коллективизаци[яи]',
    r'индустриализаци[яи]',
    r'пятилет\w*\s*план',
    r'первая\s+пятилетка',
    r'вторая\s+пятилетка',
    r'стахановск\w*',
    r'ударни[кч]\w*',
    r'социалистическ\w+\s+соревнован\w*',
    r'голод\w*\s*(?:1932|1933|голодомор)',
    r'голодомор',
]

# Procès et affaires (poids réduit à 1, car peuvent apparaître hors contexte stalinien)
TRIALS_MARKERS = [
    r'московск\w+\s+процесс',
    r'показательн\w+\s+процесс',
    r'дело\s+врач\w*',
    r'ленинградское\s+дело',
    r'антипартийн\w+\s+групп\w*',
    r'троцкист\w*',
    r'зиновьев\w*',
    r'каменев\w*',
    r'бухарин\w*',
    r'рыков\w*',
    r'тухачевск\w*',
    r'ежов\w*',
    r'берия',
    r'ягода',
]

# Années clés : ne comptent pas que si marqueur thématique présent (gating)
# Retrait de 1941/1945 qui génèrent trop de bruit (guerre sans contexte stalinien)
KEY_YEARS = [
    '1929', '1930', '1932', '1933', '1934', '1936', '1937', '1938',
    '1939', '1946', '1948', '1949', '1952', '1953'
]


def compile_patterns() -> Dict[str, re.Pattern]:
    """
    Compile tous les patterns regex pour performance.
    """
    all_patterns = {
        'stalin': STALIN_MARKERS,
        'weak_stalin': WEAK_STALIN_MARKERS,
        'cult': CULT_MARKERS,
        'repression': REPRESSION_MARKERS,
        'events': EVENT_MARKERS,
        'trials': TRIALS_MARKERS,
    }
    
    compiled = {}
    for category, patterns in all_patterns.items():
        combined = '|'.join(f'(?:{p})' for p in patterns)
        compiled[category] = re.compile(combined, re.IGNORECASE | re.UNICODE)
    
    # Pattern pour les années clés : "1937г" (sans espace avant г), "1937-м" (avec tiret),
    #"(1937)" (entre parenthèses)
    compiled['key_years'] = re.compile(r'(?<!\d)(' + '|'.join(KEY_YEARS) + r')(?!\d)', re.UNICODE)
    
    # Pattern pour toutes les années (1800-2099) pour analytique seulement
    compiled['any_year'] = re.compile(r'(?<!\d)(18\d{2}|19\d{2}|20\d{2})(?!\d)', re.UNICODE)
    
    return compiled


def normalize_yo(text: str) -> str:
    """
    Normalise ё → е pour le matching.
    """
    return text.replace("ё", "е").replace("Ё", "Е")


def count_matches(text_norm: str, patterns: Dict[str, re.Pattern]) -> Dict[str, int]:
    """
    Compte les occurrences de chaque catégorie de marqueurs.
    Utilise finditer() pour un comptage fiable.
    
    Important : any_year et key_years sont comptés séparément via findall()
    car ils sont utilisés différemment (analytics vs scoring).
    """
    counts = {}
    for category, pattern in patterns.items():
        if category in ('any_year', 'key_years'):
            continue
        counts[category] = sum(1 for _ in pattern.finditer(text_norm))
    return counts


def is_stalin_related(text_norm: str, patterns: Dict[str, re.Pattern], 
                      min_score: int = 2) -> Tuple[bool, Dict[str, int], int, bool, bool, set, str]:
    """
    Détermine si un segment est lié à Staline/période stalinienne.
    
    Système de scoring:
    - Mention directe de Staline: +3
    - Mention faible (Koba): +1
    - Marqueurs de répression/culte: +2
    - Marqueurs de procès: +1 (réduit car peuvent être hors contexte)
    - Événements de l'époque: +1
    - Dates clés: +1 (max 3 points, uniquement si marqueur thématique fort présent)
    
    GATING: Les dates ne comptent que si un marqueur thématique fort est présent.
    weak_stalin (Koba) ne suffit pas à activer les dates : il ajoute juste +1 au score.
    
    Retourne: (is_related, counts, score, has_thematic, has_strong_thematic, 
               key_years_unique, passed_reason)
    """
    counts = count_matches(text_norm, patterns)
    
    key_years_unique = set(patterns['key_years'].findall(text_norm))
    key_years_count = len(key_years_unique)
    counts['key_years'] = key_years_count
    
    # Marqueurs thématiques forts (activent le gating des dates)
    has_strong_thematic = (
        counts.get('stalin', 0) > 0
        or counts.get('cult', 0) > 0
        or counts.get('repression', 0) > 0
        or counts.get('trials', 0) > 0
        or counts.get('events', 0) > 0
    )
    
    # has_thematic inclut weak_stalin (pour les stats), mais pas pour le gating
    has_thematic = has_strong_thematic or counts.get('weak_stalin', 0) > 0
    
    WEIGHTS = {
        'stalin': 3,
        'weak_stalin': 1,
        'cult': 2,
        'repression': 2,
        'trials': 1,
        'events': 1,
    }
    
    score = 0
    contributions = {}
    for cat, weight in WEIGHTS.items():
        count = counts.get(cat, 0)
        contrib = count * weight
        contributions[cat] = contrib
        score += contrib
    
    # gating
    date_contrib = 0
    if has_strong_thematic:
        date_contrib = min(counts.get('key_years', 0), 3)
        score += date_contrib
    contributions['key_years'] = date_contrib
    
    passed_reason = None
    if counts.get('stalin', 0) > 0:
        is_related = True
        passed_reason = 'stalin_direct'
    elif has_strong_thematic and score >= min_score:
        is_related = True
        # Trouver la catégorie dominante par contribution au score
        # Exclure stalin (déjà traité), weak_stalin (pas strong), key_years (pas thématique)
        strong_categories = ['repression', 'cult', 'events', 'trials']
        
        strong_contribs = {c: contributions.get(c, 0) for c in strong_categories}
        max_contrib = max(strong_contribs.values())
        
        if max_contrib > 0:
            top_categories = [c for c, v in strong_contribs.items() if v == max_contrib]
            
            if len(top_categories) == 1:
                # Pas de tie (une seule catégorie dominante)
                passed_reason = f'strong_{top_categories[0]}'
            else:
                # Tie: plusieurs catégories à égalité
                # Tie-break par raw count (plus de marqueurs = plus spécifique)
                top_by_count = max(top_categories, key=lambda c: counts.get(c, 0))
                counts_at_top = [counts.get(c, 0) for c in top_categories]
                
                if counts_at_top.count(counts.get(top_by_count, 0)) == 1:
                    passed_reason = f'strong_{top_by_count}'
                else:
                    passed_reason = 'strong_mixed'
        else:
            # ne devrait jamais arriver si has_strong_thematic est True
            passed_reason = 'strong_unknown'
    else:
        is_related = False
    
    return is_related, counts, score, has_thematic, has_strong_thematic, key_years_unique, passed_reason


def filter_segments(segments: List[Dict], min_score: int = 2, 
                    include_all_years: bool = False) -> Tuple[List[Dict], Dict, List[Dict], List[Dict]]:
    """
    Filtre les segments pour ne garder que ceux liés à Staline.
    
    Args:
        segments: Liste des segments à filtrer
        min_score: Score minimum pour retenir un segment (défaut: 2)
        include_all_years: Si True, ajoute filter_all_years à chaque segment retenu
                          (toutes les années 1800-2099 détectées). Utile pour l'analyse
                          mais augmente la taille du JSON. Défaut: False.
    
    Retourne: (filtered_segments, statistics, rejected_date_only, all_rejected)
    """
    patterns = compile_patterns()
    filtered = []
    all_rejected = []
    
    # Segments rejetés uniquement à cause du gating (dates seules, pas de thématique)
    rejected_date_only = []
    
    # Catégories thématiques uniquement (sans any_year et key_years)
    thematic_categories = sorted([cat for cat in patterns.keys() if cat not in ('any_year', 'key_years')])
    
    stats = {
        'total_input': len(segments),
        'retained': 0,
        'by_period': {},
        'by_passed_reason': {},
        # Métriques par catégorie uniquement (sans dates)
        'by_category_segments': {cat: 0 for cat in thematic_categories},
        'by_category_matches': {cat: 0 for cat in thematic_categories},
        'score_distribution': {},
        # Statistique des années: toutes les métriques comptent les segments (pas les occurrences)
        'mentioned_key_years': {},
        'mentioned_all_years': {},
        'mentioned_all_years_rejected': {},
        # Compteurs pour les deux types de date-only
        'rejected_date_only_any': 0,    # Au moins 1 key_year sans thématique
        'rejected_date_only_strong': 0,  # >= min_score key_years sans thématique
    }
    
    for segment in segments:
        text = segment['text']
        text_norm = normalize_yo(text)
        
        is_related, counts, score, has_thematic, has_strong_thematic, key_years_unique, passed_reason = \
            is_stalin_related(text_norm, patterns, min_score)
        
        if not is_related:
            key_years_count = counts.get('key_years', 0)
            date_points = min(key_years_count, 3)

            has_weak_stalin = counts.get('weak_stalin', 0) > 0
            is_date_only_any = (not has_strong_thematic) and key_years_count > 0
            is_date_only_strong = (not has_strong_thematic) and date_points >= min_score
            
            if is_date_only_any:
                stats['rejected_date_only_any'] += 1
            if is_date_only_strong:
                stats['rejected_date_only_strong'] += 1
                
                # date_only_pure vs date_plus_weak_stalin
                if has_weak_stalin:
                    reason = 'date_plus_weak_stalin'
                else:
                    reason = 'date_only_pure'
                
                rejected_date_only.append({
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'doc_id': segment.get('doc_id'),
                    'period': segment.get('period'),
                    'key_years_count': key_years_count,
                    'date_points': date_points,
                    'has_weak_stalin': has_weak_stalin,
                    'reason': reason
                })
            
            # Garder tous les rejetés pour random sampling (avec détails pour diagnostic)
            all_rejected.append({
                'text': text[:200] + '...' if len(text) > 200 else text,
                'doc_id': segment.get('doc_id'),
                'period': segment.get('period'),
                'score': score,
                'key_years_count': key_years_count,
                'has_thematic': has_thematic,
                'has_strong_thematic': has_strong_thematic,
                # Catégories qui ont matché (pour comprendre pourquoi rejeté)
                'matched_categories': [cat for cat in thematic_categories if counts.get(cat, 0) > 0]
            })
            # Dans combien de segments rejetés est mentionnée l'année X
            for year in set(patterns['any_year'].findall(text_norm)):
                stats['mentioned_all_years_rejected'][year] = \
                    stats['mentioned_all_years_rejected'].get(year, 0) + 1
        
        if is_related:
            out_seg = dict(segment)
            out_seg['filter_score'] = score
            out_seg['filter_matches'] = counts
            out_seg['filter_key_years'] = sorted(key_years_unique)
            out_seg['filter_passed_reason'] = passed_reason
            
            all_years_unique = set(patterns['any_year'].findall(text_norm))
            
            if include_all_years:
                out_seg['filter_all_years'] = sorted(all_years_unique)
            
            filtered.append(out_seg)
            
            # Statistiques
            stats['retained'] += 1
            period = segment['period']
            stats['by_period'][period] = stats['by_period'].get(period, 0) + 1
            
            # Statistiques par raison de passage (pour validation)
            stats['by_passed_reason'][passed_reason] = \
                stats['by_passed_reason'].get(passed_reason, 0) + 1
            
            # Compter uniquement les catégories thématiques
            for cat in thematic_categories:
                count = counts.get(cat, 0)
                if count > 0:
                    stats['by_category_segments'][cat] += 1
                    stats['by_category_matches'][cat] += count
            
            # Dans combien de segments est mentionnée l'année X
            for year in key_years_unique:
                stats['mentioned_key_years'][year] = stats['mentioned_key_years'].get(year, 0) + 1
            
            for year in all_years_unique:
                stats['mentioned_all_years'][year] = stats['mentioned_all_years'].get(year, 0) + 1
            
            score_bucket = f"{(score // 5) * 5}-{(score // 5) * 5 + 4}"
            stats['score_distribution'][score_bucket] = \
                stats['score_distribution'].get(score_bucket, 0) + 1
    
    stats['retention_rate'] = round(stats['retained'] / stats['total_input'] * 100, 2) \
        if stats['total_input'] > 0 else 0.0
    stats['rejected_date_only_count'] = len(rejected_date_only)
    stats['total_rejected'] = len(all_rejected)
    
    return filtered, stats, rejected_date_only, all_rejected


def process_filtering(input_path: Path, output_path: Path, min_score: int = 2,
                      include_all_years: bool = False) -> Dict:
    """
    Charge les segments, filtre et sauvegarde.
    Génère également un rapport de qualité pour vérification manuelle.
    
    Args:
        input_path: Chemin vers le fichier JSON des segments
        output_path: Chemin pour le fichier JSON filtré
        min_score: Score minimum pour retenir un segment
        include_all_years: Si True, ajoute filter_all_years à chaque segment
    """
    print(f"Chargement des segments depuis {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    print(f"Filtrage avec score minimum = {min_score}...")
    if include_all_years:
        print("  (include_all_years=True: filter_all_years sera ajouté aux segments)")
    filtered, stats, rejected_date_only, all_rejected = filter_segments(
        segments, min_score, include_all_years
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les segments filtrés
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"Segments filtrés exportés: {output_path}")
    
    # Sauvegarder les statistiques séparément
    stats_path = output_path.parent / "filtering_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistiques exportées: {stats_path}")
    
    quality_report = {
        'description': 'Échantillons pour vérification manuelle du filtrage',
        'min_score_used': min_score,
        
        # Top 50 segments avec le score le plus élevé
        'top_scoring': [],
        
        # 50 segments "sur la frontière" (score == min_score)
        'borderline': [],
        
        # 50 segments rejetés car ils n'avaient que des dates (pas de marqueurs thématiques)
        'rejected_date_only_samples': rejected_date_only[:50],
        
        # Échantillons aléatoires pour évaluer la précision moyenne
        'random_retained': [],
        'random_rejected': [],
        
        # Top années dans les segments rejetés (pour diagnostiquer le bruit)
        'top_years_in_rejected': sorted(
            stats['mentioned_all_years_rejected'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20],
    }
    
    # Trier par score décroissant
    sorted_by_score = sorted(filtered, key=lambda x: x['filter_score'], reverse=True)
    
    # Top scoring
    for seg in sorted_by_score[:50]:
        quality_report['top_scoring'].append({
            'text': seg['text'][:300] + '...' if len(seg['text']) > 300 else seg['text'],
            'doc_id': seg.get('doc_id'),
            'period': seg.get('period'),
            'score': seg['filter_score'],
            'matches': seg['filter_matches']
        })
    
    # Borderline (score == min_score)
    borderline = [s for s in filtered if s['filter_score'] == min_score]
    for seg in borderline[:50]:
        quality_report['borderline'].append({
            'text': seg['text'][:300] + '...' if len(seg['text']) > 300 else seg['text'],
            'doc_id': seg.get('doc_id'),
            'period': seg.get('period'),
            'score': seg['filter_score'],
            'matches': seg['filter_matches']
        })
    
    # Random samples (retained)
    random_retained_sample = random.sample(filtered, min(50, len(filtered))) if filtered else []
    for seg in random_retained_sample:
        quality_report['random_retained'].append({
            'text': seg['text'][:300] + '...' if len(seg['text']) > 300 else seg['text'],
            'doc_id': seg.get('doc_id'),
            'period': seg.get('period'),
            'score': seg['filter_score'],
            'matches': seg['filter_matches']
        })
    
    # Random samples (rejected)
    random_rejected_sample = random.sample(all_rejected, min(50, len(all_rejected))) if all_rejected else []
    quality_report['random_rejected'] = random_rejected_sample
    
    quality_report['counts'] = {
        'top_scoring_shown': len(quality_report['top_scoring']),
        'borderline_total': len(borderline),
        'borderline_shown': len(quality_report['borderline']),
        'rejected_date_only_total': len(rejected_date_only),
        'rejected_date_only_shown': len(quality_report['rejected_date_only_samples']),
        'random_retained_shown': len(quality_report['random_retained']),
        'random_rejected_shown': len(quality_report['random_rejected']),
        'total_retained': len(filtered),
        'total_rejected': len(all_rejected),
    }
    
    quality_path = output_path.parent / "filtering_quality_report.json"
    with open(quality_path, 'w', encoding='utf-8') as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)
    print(f"Rapport qualité exporté: {quality_path}")
    
    return stats


if __name__ == '__main__':
    import sys
    
    # Seed pour reproductibilité des échantillons aléatoires
    random.seed(42)
    
    args = sys.argv[1:]
    include_all_years = '--all-years' in args
    if include_all_years:
        args.remove('--all-years')
    
    if len(args) < 2:
        print("Usage: python step3_filter.py <input_segments.json> <output_filtered.json> [min_score] [--all-years]")
        print("Exemple: python step3_filter.py ./segments_all.json ./segments_stalin.json 2")
        print("         python step3_filter.py ./segments_all.json ./segments_stalin.json 2 --all-years")
        print("")
        print("Options:")
        print("  --all-years  Ajouter filter_all_years (toutes les années 1800-2099) aux segments")
        sys.exit(1)
    
    input_path = Path(args[0])
    output_path = Path(args[1])
    min_score = int(args[2]) if len(args) > 2 else 2
    
    stats = process_filtering(input_path, output_path, min_score, include_all_years)
    
    print(f"\n{'='*50}")
    print(f"Filtrage terminé")
    print(f"  Segments en entrée: {stats['total_input']}")
    print(f"  Segments retenus: {stats['retained']} ({stats['retention_rate']}%)")
    print(f"  Segments rejetés: {stats['total_rejected']}")
    print(f"  Rejetés date-only (any): {stats['rejected_date_only_any']}")
    print(f"  Rejetés date-only (strong, >= {min_score} pts): {stats['rejected_date_only_strong']}")
    print(f"\nPar raison de passage (validation):")
    for reason, count in sorted(stats['by_passed_reason'].items(), key=lambda x: -x[1]):
        pct = round(count / stats['retained'] * 100, 1) if stats['retained'] > 0 else 0
        print(f"  {reason:<20} {count:>8} ({pct}%)")
    print(f"\nPar période:")
    for period, count in sorted(stats['by_period'].items()):
        print(f"  {period}: {count} segments")
    print(f"\nPar catégorie de marqueurs:")
    print(f"  {'Catégorie':<15} {'Segments':>10} {'Matches':>10}")
    print(f"  {'-'*35}")
    for cat in sorted(stats['by_category_segments'].keys()):
        seg_count = stats['by_category_segments'][cat]
        match_count = stats['by_category_matches'][cat]
        print(f"  {cat:<15} {seg_count:>10} {match_count:>10}")
    print(f"\nDistribution des scores:")
    for bucket, count in sorted(stats['score_distribution'].items()):
        print(f"  {bucket}: {count}")
    print(f"\nAnnées clés les plus mentionnées (KEY_YEARS) - retenus:")
    sorted_key_years = sorted(stats['mentioned_key_years'].items(), key=lambda x: x[1], reverse=True)
    for year, count in sorted_key_years[:10]:
        print(f"  {year}: {count}")
    print(f"\nToutes les années les plus mentionnées - retenus:")
    sorted_all_years = sorted(stats['mentioned_all_years'].items(), key=lambda x: x[1], reverse=True)
    for year, count in sorted_all_years[:15]:
        print(f"  {year}: {count}")
    print(f"\nAnnées les plus mentionnées - REJETÉS (diagnostic du bruit):")
    sorted_rejected_years = sorted(stats['mentioned_all_years_rejected'].items(), key=lambda x: x[1], reverse=True)
    for year, count in sorted_rejected_years[:10]:
        print(f"  {year}: {count}")
