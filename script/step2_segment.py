"""
Étape 2 : Segmentation du corpus en unités d'analyse
1/ Découpage en paragraphes (unité principale)
2/ Subdivision des paragraphes longs en segments de 1-5 phrases
  (objectif: 2-5 phrases pour les paragraphes longs, mais les paragraphes
   courts d'une seule phrase sont conservés pour préserver la granularité)
3/ Préservation des métadonnées (source, période, année, niveau, espace narratif)
"""

import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple


# Limites pour SBERT
MAX_SEGMENT_CHARS = 1500  # 300-400 tokens pour le russe
MIN_SEGMENT_WORDS = 15    # Segments trop courts = peu informatifs
MAX_SENTENCES = 5         # Maximum 5 phrases par segment
MIN_PARAGRAPH_WORDS = 8   # Paragraphes trop courts = probablement des titres


@dataclass
class Segment:
    """Unité d'analyse avec métadonnées."""
    text: str
    doc_id: str
    period: str
    paragraph_idx: int
    segment_idx: int
    source_file: str
    char_count: int
    word_count: int
    sentence_count: int
    # Métadonnées du document (depuis metadata.csv)
    year: Optional[int] = None
    level: Optional[str] = None
    country: Optional[str] = None


# Abréviations russes courantes
# Utilisation de délimiteurs (pour éviter les collisions avec le texte réel)
# Important: triées par longueur décroissante pour éviter les remplacements partiels
# ("гг." doit être remplacé avant "г.")
ABBREVIATIONS = sorted([
    ("г.", "__YEAR_ABBR__"),
    ("гг.", "__YEARS_ABBR__"),
    ("в.", "__CENTURY_ABBR__"),
    ("вв.", "__CENTURIES_ABBR__"),
    ("т.д.", "__ETC_ABBR__"),
    ("т.п.", "__ETC2_ABBR__"),
    ("т.е.", "__IE_ABBR__"),
    ("т.н.", "__SOCALLED_ABBR__"),
    ("т.к.", "__BECAUSE_ABBR__"),
    ("др.", "__OTHER_ABBR__"),
    ("см.", "__SEE_ABBR__"),
    ("ср.", "__COMPARE_ABBR__"),
    ("тыс.", "__THOUSAND_ABBR__"),
    ("млн.", "__MILLION_ABBR__"),
    ("млрд.", "__BILLION_ABBR__"),
    ("руб.", "__RUB_ABBR__"),
    ("чел.", "__PERSON_ABBR__"),
    ("проч.", "__OTHERS_ABBR__"),
    ("напр.", "__EXAMPLE_ABBR__"),
    ("ок.", "__CIRCA_ABBR__"),
    ("им.", "__NAMED_ABBR__"),
    ("акад.", "__ACAD_ABBR__"),
    ("проф.", "__PROF_ABBR__"),
], key=lambda x: len(x[0]), reverse=True)


def count_words(text: str) -> int:
    """
    Compte le nombre de mots dans le texte.
    Utilise une regex pour une tokenisation plus précise:
    1/ Gère correctement les tirets (1937–1938 = 2 tokens)
    2/ Gère les abréviations avec points
    3/ Compte les nombres comme des mots
    """
    return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text))


def split_into_sentences(text: str) -> List[str]:
    """
    Découpe le texte en phrases.
    Gère les abréviations russes courantes et les guillemets/parenthèses.
    """
    # Protéger les abréviations
    protected = text
    for abbr, placeholder in ABBREVIATIONS:
        protected = re.sub(re.escape(abbr), placeholder, protected, flags=re.IGNORECASE)
    
    # Découper sur les fins de phrase
    # Gère: ." .) !" ?» etc.
    sentences = re.split(r'(?:(?<=[.!?])|(?<=[.!?][»")\]]))\s+', protected)
    
    # Restaurer les abréviations
    result = []
    for sent in sentences:
        restored = sent
        for abbr, placeholder in ABBREVIATIONS:
            restored = restored.replace(placeholder, abbr)
        if restored.strip():
            result.append(restored.strip())
    
    return result


def split_into_paragraphs(text: str) -> List[str]:
    """
    Découpe le texte en paragraphes.
    Un paragraphe = bloc séparé par une ligne vide.
    Filtre par nombre de mots (pas par caractères).
    """
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs 
            if p.strip() and count_words(p.strip()) >= MIN_PARAGRAPH_WORDS]


def segment_paragraph(
    paragraph: str, 
    max_chars: int = MAX_SEGMENT_CHARS
) -> List[Tuple[str, int]]:
    """
    Si le paragraphe est trop long, le subdivise en segments de 2-5 phrases.
    Garantit un maximum de MAX_SENTENCES phrases par segment.
    
    Returns:
        Liste de tuples (texte_segment, nombre_de_phrases)
        Retourner le nombre de phrases évite de recalculer split_into_sentences()
    
    Les segments d'une phrase sont acceptés pour:
    1/ Les paragraphes courts (≤5 phrases, ≤max_chars)
    2/ Le dernier segment si fusion impossible
    """

    sentences = split_into_sentences(paragraph)
    num_sentences = len(sentences)
    
    # Paragraphe court: retourner tel quel
    if len(paragraph) <= max_chars and num_sentences <= MAX_SENTENCES:
        return [(paragraph, num_sentences)]
    
    # Paragraphe long mais peu de phrases: garder tel quel
    if num_sentences <= 2:
        return [(paragraph, num_sentences)]
    
    # Subdiviser en segments
    segments: List[Tuple[str, int]] = []
    current_segment: List[str] = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # Couper si: 1/ on dépasse max_chars ET on a ≥2 phrases
        # ou 2/ on a atteint MAX_SENTENCES phrases
        should_cut = (
            (current_length + sentence_len > max_chars and len(current_segment) >= 2)
            or len(current_segment) >= MAX_SENTENCES
        )
        
        if should_cut:
            segments.append((' '.join(current_segment), len(current_segment)))
            current_segment = [sentence]
            current_length = sentence_len
        else:
            current_segment.append(sentence)
            current_length += sentence_len + 1  # +1 pour l'espace
    
    # Ajouter le dernier segment
    if current_segment:
        # Si le dernier segment est trop court (1 phrase), essayer de fusionner
        if len(current_segment) == 1 and segments:
            last_text, last_count = segments[-1]
            if last_count < MAX_SENTENCES:
                # Fusionner avec le segment précédent
                segments[-1] = (last_text + ' ' + current_segment[0], last_count + 1)
            else:
                # Impossible de fusionner: accepter le segment d'une phrase
                segments.append((' '.join(current_segment), len(current_segment)))
        else:
            segments.append((' '.join(current_segment), len(current_segment)))
    
    return segments


def process_document(
    text: str, 
    doc_id: str, 
    period: str, 
    source_file: str,
    year: Optional[int] = None,
    level: Optional[str] = None,
    country: Optional[str] = None
) -> List[Segment]:
    """
    Traite un document complet et retourne la liste des segments.
    """
    paragraphs = split_into_paragraphs(text)
    segments = []
    
    for para_idx, paragraph in enumerate(paragraphs):
        sub_segments = segment_paragraph(paragraph)
        
        for seg_idx, (seg_text, sentence_count) in enumerate(sub_segments):
            word_count = count_words(seg_text)
            
            # Filtrer les segments trop courts
            if word_count < MIN_SEGMENT_WORDS:
                continue
            
            segment = Segment(
                text=seg_text,
                doc_id=doc_id,
                period=period,
                paragraph_idx=para_idx,
                segment_idx=seg_idx,
                source_file=source_file,
                char_count=len(seg_text),
                word_count=word_count,
                sentence_count=sentence_count,
                year=year,
                level=level,
                country=country
            )
            segments.append(segment)
    
    return segments


def load_metadata(metadata_path: Path) -> Dict[str, dict]:
    """
    Charge les métadonnées depuis le fichier CSV.
    Retourne un dict: doc_id -> metadata
    """
    metadata = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['doc_id']] = row
    return metadata


def get_metadata_value(meta: dict, *keys, convert_int: bool = False):
    """
    Récupère une valeur depuis les métadonnées avec fallback sur plusieurs clés.
    Permet de gérer différentes conventions de nommage (FR/EN/RU).
    
    Args:
        meta: Dictionnaire de métadonnées
        *keys: Clés à essayer dans l'ordre
        convert_int: Si True, convertit en int
    
    Returns:
        La valeur trouvée ou None
    """
    for key in keys:
        value = meta.get(key)
        if value is not None and value != '':
            if convert_int:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return None
            return value
    return None


def process_corpus(corpus_dir: Path, metadata_path: Path, output_dir: Path) -> dict:
    """
    Traite le corpus entier et sauvegarde les segments en JSON et CSV.
    """
    metadata = load_metadata(metadata_path)
    all_segments = []
    stats_by_period = {}
    missing_metadata = []
    
    for period_dir in sorted(corpus_dir.iterdir()):
        if not period_dir.is_dir() or period_dir.name.startswith('.'):
            continue
        
        period = period_dir.name
        period_segments = []
        print(f"\nTraitement de la période: {period}")
        
        for txt_file in sorted(period_dir.glob('*.txt')):
            # Extraire doc_id du nom de fichier (ex: berkhin_1980.txt -> berkhin_1980)
            doc_id = txt_file.stem
            
            # Récupérer les métadonnées avec fallback keys
            meta = metadata.get(doc_id, {})
            
            if not meta:
                missing_metadata.append(doc_id)
            
            year = get_metadata_value(meta, 'year', 'annee', 'год', convert_int=True)
            
            level = get_metadata_value(meta, 'level', 'niveau', 'уровень')
            
            country = get_metadata_value(meta, 'country', 'pays', 'espace_narratif', 'страна')
            
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            segments = process_document(
                text, doc_id, period, txt_file.name,
                year=year, level=level, country=country
            )
            period_segments.extend(segments)
            
            meta_status = "✓" if meta else "(no metadata)"
            print(f"  {meta_status} {doc_id}: {len(segments)} segments")
        
        if period_segments:
            stats_by_period[period] = {
                'num_segments': len(period_segments),
                'total_chars': sum(s.char_count for s in period_segments),
                'total_words': sum(s.word_count for s in period_segments),
                'avg_segment_words': round(sum(s.word_count for s in period_segments) / len(period_segments), 1)
            }
        all_segments.extend(period_segments)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / "segments_all.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(s) for s in all_segments], f, ensure_ascii=False, indent=2)
    print(f"\nJSON exporté: {json_path}")
    
    # Sauvegarde en CSV (plus pratique pour embeddings/clustering)
    csv_path = output_dir / "segments_all.csv"
    if all_segments:
        fieldnames = list(asdict(all_segments[0]).keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for seg in all_segments:
                writer.writerow(asdict(seg))
    print(f"CSV exporté: {csv_path}")
    
    # Les statistiques
    stats_path = output_dir / "segmentation_stats.json"
    stats = {
        'total_segments': len(all_segments),
        'total_words': sum(s.word_count for s in all_segments),
        'missing_metadata_doc_ids': missing_metadata,
        'by_period': stats_by_period
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Stats exportées: {stats_path}")
    
    if missing_metadata:
        print(f"\n ATTENTION: {len(missing_metadata)} documents sans métadonnées:")
        for doc_id in missing_metadata[:5]:
            print(f"    - {doc_id}")
        if len(missing_metadata) > 5:
            print(f"    ... et {len(missing_metadata) - 5} autres")
    
    return stats


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 step2_segment.py <corpus_dir> <metadata.csv> <output_dir>")
        sys.exit(1)
    
    corpus_dir = Path(sys.argv[1])
    metadata_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    
    stats = process_corpus(corpus_dir, metadata_path, output_dir)
    
    print(f"\n{'='*50}")
    print(f"Segmentation terminée: {stats['total_segments']} segments")
    print(f"Total mots: {stats['total_words']:,}")
    print("\nPar période:")
    for period, pstats in stats['by_period'].items():
        print(f"  {period}: {pstats['num_segments']} segments, "
              f"~{pstats['avg_segment_words']} mots/segment")
