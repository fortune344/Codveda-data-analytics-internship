# Codveda Data Analytics Internship — Level 1 (Basic)

Projet réalisé dans le cadre du stage **Data Analytics** chez **[Codveda Technology](https://www.codveda.com)**.

Ce dépôt couvre les **Tasks 1 & 2** du **Level 1 (Basic)**, appliquées au dataset **Iris**.

##  Tâches réalisées

### Task 1 — Data Cleaning and Preprocessing
- Chargement du dataset avec `pandas`
- Détection et traitement des valeurs manquantes
- Détection et suppression des doublons (3 lignes dupliquées trouvées et supprimées : 150 → 147 lignes)
- Standardisation des formats incohérents (casse et espaces de la colonne `species`)

### Task 2 — Exploratory Data Analysis (EDA)
- Statistiques descriptives (moyenne, médiane, mode, écart-type), globales et par espèce
- Visualisation des distributions : histogrammes, boxplots par espèce
- Relations entre variables : pairplot (scatter plots croisés)
- Matrice de corrélation entre variables numériques

##  Principaux insights

- Aucune valeur manquante dans le dataset ; 3 doublons supprimés lors du nettoyage.
- `petal_length` et `petal_width` sont très fortement corrélées (r ≈ 0.96) — ce sont les deux variables les plus discriminantes entre espèces.
- **Setosa** se distingue nettement des deux autres espèces sur les variables liées aux pétales (aucun chevauchement).
- **Versicolor** et **virginica** se chevauchent partiellement, virginica ayant globalement des pétales et sépales plus grands.
- Le dataset nettoyé est prêt pour une modélisation (classification, clustering) en Level 2/3.

##  Outils utilisés

- Python
- pandas
- matplotlib
- seaborn

##  Structure du dépôt

```
├── Codveda_Iris_Task1_Task2.ipynb   # Notebook principal (Task 1 + Task 2)
├── iris.csv                          # Dataset brut
├── iris_clean.csv                    # Dataset nettoyé (sortie de la Task 1)
└── README.md
```

##  Utilisation

```bash
pip install pandas matplotlib seaborn jupyter
jupyter notebook Codveda_Iris_Task1_Task2.ipynb
```

