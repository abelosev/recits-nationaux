"""
Étape 1 : Normalisation légère des données textuelles
1/ Unification des espaces et des tirets
2/ Suppression des numéros de pages isolés
3/ Suppression des renvois bibliographiques ([1], [см. 5]) et indications de page (стр. 45)
"""

import csv
import re
import sys
from pathlib import Path


def normalize_text(text: str) -> str:
    """
    Normalisation légère du texte brut.
    Préserve la structure discursive tout en nettoyant les artefacts.
    """
    
    # 1/ Normalisation des fins de ligne (Windows → Unix)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 2/ Normalisation des espaces spéciaux (NBSP fréquent dans les PDF)
    text = text.replace('\u00a0', ' ')  # non-breaking space
    text = text.replace('\u200b', '')   # zero-width space
    text = text.replace('\f', '\n\n')   # form feed (page break) → paragraph break
    
    # 3/ Unification des tirets (différents types → tiret standard)
    text = re.sub(r'[‐‑‒–—―]', '-', text)
    
    # 4/ Unification des espaces (multiples espaces, tabs, etc.)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 5/ Suppression des numéros de pages isolés
    #    Pattern: lignes contenant uniquement des chiffres (± tirets/espaces)
    #    Multiligne pour attraper début/fin de fichier aussi
    text = re.sub(r'(?m)^\s*-?\s*\d{1,4}\s*-?\s*$', '', text)
    
    # 6/ Suppression des références de pages (стр. 45, Стр. 45, стр 45)
    text = re.sub(r'\bстр\.?\s*\d{1,4}\b', '', text, flags=re.IGNORECASE)
    
    # 7/ Suppression des références bibliographiques entre crochets
    #    Pattern: [1], [12], [см. 5], [1, 2, 3], [1-3], [1–5] etc.
    #    Limite stricte à 1-3 chiffres pour ne pas supprimer [1937], [1953]
    text = re.sub(
        r'\[\s*(?:см\.?\s*)?\d{1,3}(?:\s*[-–—]\s*\d{1,3})?(?:\s*[,;]\s*\d{1,3})*\s*\]',
        '',
        text
    )
    
    # 8/ Normalisation des sauts de ligne (max 2 consécutifs)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 9/ Fusion des mots coupés par césure (fréquent dans les PDF)
    #    Ex: индустриа-\nлизация → индустриализация
    text = re.sub(r'([A-Za-zА-Яа-яЁё])-\s*\n\s*([A-Za-zА-Яа-яЁё])', r'\1\2', text)
    
    # 10/ Fusion des retours à la ligne à l'intérieur des paragraphes
    #     Préserve les paragraphes (double \n), fusionne les lignes simples
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 11/ Re-normalisation des espaces après fusion
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 12/ Suppression des espaces en début/fin de ligne
    text = '\n'.join(line.strip() for line in text.split('\n'))
    
    # 13/ Heuristique pour textes OCR mal formatés (sans paragraphes)
    #     Si le texte a très peu de sauts de ligne, on essaie de recréer la structure
    #     en ajoutant des sauts après les points suivis de majuscules
    #     (condition stricte pour éviter les faux positifs)
    if text.count('\n') < 20 and len(text) > 10000:
        text = re.sub(r'(?<![тТгГ])\.([А-ЯЁA-Z])', r'.\n\n\1', text)
        text = re.sub(r'§\s*(\d+)', r'\n\n§ \1', text)
    
    return text.strip()


def process_file(input_path: Path, output_path: Path) -> dict:
    """
    Traite un fichier et retourne des statistiques détaillées.
    """
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        original_text = f.read()
    
    normalized_text = normalize_text(original_text)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(normalized_text)
    
    # Statistiques détaillées
    original_lines = original_text.count('\n') + 1
    normalized_lines = normalized_text.count('\n') + 1
    normalized_paragraphs = len([p for p in normalized_text.split('\n\n') if p.strip()])
    
    return {
        'file': input_path.name,
        'original_chars': len(original_text),
        'normalized_chars': len(normalized_text),
        'reduction_pct': round((1 - len(normalized_text)/len(original_text)) * 100, 2) if original_text else 0,
        'original_lines': original_lines,
        'normalized_lines': normalized_lines,
        'paragraphs': normalized_paragraphs
    }


def process_corpus(corpus_dir: Path, output_dir: Path) -> list:
    """
    Traite tous les fichiers du corpus organisé par périodes.
    """
    stats = []
    
    # Parcourir les sous-dossiers
    for period_dir in sorted(corpus_dir.iterdir()):
        if period_dir.is_dir() and not period_dir.name.startswith('.'):
            print(f"\nTraitement de la période: {period_dir.name}")
            
            for txt_file in sorted(period_dir.glob('*.txt')):
                output_path = output_dir / period_dir.name / txt_file.name
                file_stats = process_file(txt_file, output_path)
                file_stats['period'] = period_dir.name
                stats.append(file_stats)
                print(f"{txt_file.name}: {file_stats['reduction_pct']}% réduit")
    
    return stats


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python step1_normalize.py <corpus_dir> <output_dir>")
        sys.exit(1)
    
    corpus_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    stats = process_corpus(corpus_dir, output_dir)
    
    print(f"\n{'='*50}")
    print(f"Normalisation terminée: {len(stats)} fichiers traités")
    total_original = sum(s['original_chars'] for s in stats)
    total_normalized = sum(s['normalized_chars'] for s in stats)
    total_paragraphs = sum(s['paragraphs'] for s in stats)
    if total_original:
        print(f"Réduction totale: {round((1 - total_normalized/total_original) * 100, 2)}%")
    else:
        print("Réduction totale: n/a (corpus vide)")
    print(f"Paragraphes préservés: {total_paragraphs}")
    
    # Export des statistiques en CSV pour les annexes
    if stats:
        out_stats = output_dir / "normalization_stats.csv"
        with open(out_stats, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)
        print(f"Stats exportées: {out_stats}")
