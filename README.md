# Analyse des récits nationaux sur Staline dans les manuels d'histoire russes

Analyse diachronique de 22 manuels (1947–2023) combinant méthodes de TAL (embeddings SBERT, clustering HDBSCAN).

## Structure du projet

```
corpus/                     # Corpus de manuels (textes bruts)
├── 1947-1953/
├── ...
├── 2013-2023/
└── metadata.csv            # Métadonnées (auteur, année, niveau)

script/                     # Pipeline de traitement
├── run_pretraitement.py    # Lance les étapes 1-4
├── step1_normalize.py      # Nettoyage du texte
├── step2_segment.py        # Segmentation en paragraphes
├── step3_filter.py         # Filtrage thématique (Staline)
├── step4_dedup.py          # Suppression des doublons
├── step5_embeddings.py     # Vectorisation (SBERT)
├── step6_cluster.py        # Clustering (UMAP + HDBSCAN)
└── visualization.py        # Graphiques
```

## Démarrage rapide

```bash
# 1. Installation des dépendances
pip install -r requirements.txt

# 2. Prétraitement (étapes 1-4)
python3 script/run_pretraitement.py corpus/ pretraitement_out/

# 3. Génération des embeddings
python3 script/step5_embeddings.py pretraitement_out/04_dedup/segments_final.json embeddings_out/

# 4. Clusterisation
python3 script/step6_cluster.py embeddings_out/ clustering_out/

# 5. Visualisation
python3 script/visualization.py clustering_out/ figures/
```

## Paramètres principaux

| Étape | Paramètre | Valeur |
|-------|-----------|--------|
| Filtrage | min_score | 2 |
| Déduplication | fuzzy_threshold | 0.85 |
| Embeddings | modèle | `ai-forever/sbert_large_nlu_ru` |
| UMAP | n_neighbors / metric | 15 / cosine |
| HDBSCAN | min_cluster_size | 10 |

## Sorties

- `segments_final.json` — 3 528 segments filtrés
- `clustering_out/` — labels des clusters, projections UMAP
- `figures/` — heatmap, UMAP, histogrammes

## Prérequis

- sentence-transformers, umap-learn, hdbscan, scikit-learn, pandas, matplotlib