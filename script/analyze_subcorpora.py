#!/usr/bin/env python3
"""
python analyze_subcorpora.py <PATH to corpus> <PATH to metadata.csv>
"""
import pandas as pd
import os
import sys
from collections import defaultdict

def count_words_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            words = text.split()
            return len(words)
    except Exception as e:
        print(f"Erreur lors de la lecture de {file_path}: {e}")
        return 0

def analyze_subcorpora(corpus_path, metadata_path):
    # Dictionnaire de traduction des noms de pays
    country_translation = {
        'СССР': 'URSS',
        'CCCР': 'URSS',
        'СССР(Эстония)': "RSS d'Estonie",
        'СССР(Молдавия)': 'RSS moldave',
        'СССР(Украина)': "RSS d'Ukraine",
        'СССР(Казахстан)': 'RSS kazakhe',
        'Россия': 'Russie'
    }
    
    metadata = pd.read_csv(metadata_path)
    doc_to_country = dict(zip(metadata['doc_id'], metadata['pays']))
    
    subcorpora = defaultdict(list)
    
    for item in os.listdir(corpus_path):
        item_path = os.path.join(corpus_path, item)
        if os.path.isdir(item_path) and item != '__pycache__':
            period = item
            for file in os.listdir(item_path):
                if file.endswith('.txt'):
                    doc_id = file.replace('.txt', '')
                    subcorpora[period].append(doc_id)
    
    results = []
    grand_total_words = 0
    
    for period in sorted(subcorpora.keys()):
        print(f"\nPériode: {period}")
        doc_ids = subcorpora[period]
        total_textbooks = len(doc_ids)
        total_words = 0
        countries = set()
        
        for doc_id in doc_ids:
            if doc_id in doc_to_country:
                country = doc_to_country[doc_id]
                translated_country = country_translation.get(country, country)
                countries.add(translated_country)
            
            file_path = os.path.join(corpus_path, period, f"{doc_id}.txt")
            
            if os.path.exists(file_path):
                words = count_words_in_file(file_path)
                total_words += words
                print(f"    {doc_id}: {words:,} mots")
            else:
                print(f"    {doc_id}: fichier introuvable")
        
        grand_total_words += total_words
        
        countries_list = ', '.join(sorted(countries))
        
        results.append({
            'Période': period,
            'Nombre de manuels': total_textbooks,
            'Nombre de mots': total_words,
            'Espace narratif': countries_list
        })
        
        print(f"  Total pour {period}: {total_words:,} mots")
    
    print("\n" + "=" * 60)
    print(f"NOMBRE TOTAL DE MOTS DANS L'ENSEMBLE DU CORPUS : {grand_total_words:,}")
    print("=" * 60 + "\n")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Période')
    
    output_path = 'subcorpora_statistics.csv'
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Statistiques enregistrées dans : {output_path}")
    
    return df_results, grand_total_words


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_subcorpora.py <PATH to corpus> <PATH to metadata.csv>")
        sys.exit(1)
    
    corpus_path = sys.argv[1]
    metadata_path = sys.argv[2]
    
    if not os.path.exists(corpus_path):
        print(f"Erreur: Le chemin du corpus n'existe pas: {corpus_path}")
        sys.exit(1)
    
    if not os.path.exists(metadata_path):
        print(f"Erreur: Le fichier metadata n'existe pas: {metadata_path}")
        sys.exit(1)
    
    df, total_words = analyze_subcorpora(corpus_path, metadata_path)