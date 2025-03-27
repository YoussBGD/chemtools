# Outils d'Analyse Moléculaire

Ce dépôt contient deux outils complémentaires pour l'analyse et la comparaison de structures moléculaires. Ces applications web, développées avec Streamlit et RDKit, permettent d'explorer et visualiser des ensembles de molécules de façon interactive.

## Description des outils

### 1. Recherche analogues ciblés (`find_modif.py`)

Cet outil permet d'identifier des analogues moléculaires dans lesquels une structure centrale (scaffold) reste chimiquement identique tandis que des fragments spécifiques varient.

**Fonctionnalités principales:**
- Identification automatique des fragments moléculaires (cycles, chaînes, groupes fonctionnels)
- Sélection interactive des parties des analogues différents de la molécule référence I0
- Visualisation des modifications par rapport à la molécule de référence
- Export des résultats au format CSV avec toutes les données originales

### 2. Comparaison avec la moélcule référence I0 (`comp_I0_V2.py`)

Cet outil permet de comparer visuellement une molécule de référence (I0) avec d'autres molécules analogues et d'identifier leurs similitudes et différences structurelles.

**Fonctionnalités principales:**
- Navigation intuitive dans une bibliothèque de molécules
- Alignement automatique des molécules pour une comparaison optimale
- Mise en évidence des différences structurelles par code couleur
- Identification et visualisation des molécules similaires aux analogues parmi celles déjà sélectionnées
- Export des molécules sélectionnées au format CSV

## Installation

### Créer un nouvel environnement
```bash
conda create -n chemtools python=3.10
conda activate chemtools
```

### Installer les dépendances
```bash
conda install -c conda-forge rdkit=2023.3.3 numpy=1.25.2 pandas=2.2.3 matplotlib=3.10.1 pillow=11.1.0
pip install streamlit==1.43.0
```

## Format des données d'entrée

Les deux applications attendent un fichier CSV (ou TSV pour `find_modif.py`) contenant au minimum les colonnes suivantes:
- `SMILES`: Notation SMILES des molécules
- `ID`: Identifiant unique de chaque molécule

La molécule de référence I0 doit avoir l'ID `Molport-001-492-296`.

## Utilisation

### Recherche analogues ciblés :
1. Lancez l'application: `streamlit run find_modif.py`
2. Chargez votre fichier CSV/TSV contenant les molécules
3. Sélectionnez les fragments que vous souhaitez voir modifiés
4. Validez le scaffold invariant généré automatiquement
5. Lancez la recherche d'analogues
6. Sélectionnez et exportez les molécules d'intérêt

### Comparaison avec I0:
1. Lancez l'application: `streamlit run comp_I0_V2.py`
2. Chargez votre fichier CSV contenant les molécules
3. Naviguez entre les différentes molécules via l'interface
4. Utilisez les outils d'alignement, rotation et symétrie pour optimiser la comparaison
5. Observez les différences structurelles mises en évidence
6. Sélectionnez les molécules d'intérêt et exportez-les

## Limitations connues
- Le traitement de très grandes bibliothèques de molécules peut être lent
