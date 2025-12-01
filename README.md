# Brain Tumor Semi-Supervised Detection
# BrainScanAI – Détection de tumeurs cérébrales (Semi-supervisé)

    Projet réalisé dans le cadre de ma formation Data Scientist / Machine Learning, Mission 7 : Analyse d’images médicales avec des méthodes semi-supervisées.

    Ce projet consiste à explorer un jeu de données d’IRM cérébrales, extraire des caractéristiques visuelles via un modèle pré-entraîné, puis appliquer des méthodes de clustering et d’apprentissage semi-supervisé pour identifier et prédire la présence de tumeurs.

## 📁 Structure du projet

    brain-tumor-semisupervised/
    ├── data/                          # Données non versionnées (voir .gitignore)
    │   └── mri_dataset_brain_cancer_oc/
    │       ├── avec_labels/
    │       │   ├── cancer/
    │       │   └── normal/
    │       └── sans_labels/
    ├── notebooks/
    │   ├── 01_exploration_donnees.ipynb
    │   ├── 02_extraction_features.ipynb
    │   └── 03_semi_supervised.ipynb
    ├── src/
    │   ├── __init__.py
    │   ├── utils.py / eda_utils.py           # Fonctions d’EDA tabulaire
    │   └── image_utils.py                    # Fonctions d’EDA images et loaders
    ├── reports/
    │   └── presentation_brainscanai.pdf      # Support de présentation final
    ├── requirements.txt
    ├── README.md
    └── .gitignore

## 🧠 Objectifs du projet
    ### Étape 1 — Exploration des données

        Charger les images annotées et non annotées

        Vérifier la structure du dataset

        Analyser la résolution, mode couleur, qualité

        Visualiser un échantillon d’images

        Construire un DataFrame synthétique

    ### Étape 2 — Extraction des caractéristiques

        Charger un modèle pré-entraîné (ResNet, DenseNet201…)

        Générer et sauvegarder les embeddings

        Étudier la distribution des embeddings

        Réduire la dimension (PCA, t-SNE, UMAP)

    ### Étape 3 — Clustering

        Appliquer K-Means, DBSCAN ou HDBSCAN

        Évaluer les clusters vs labels connus

        Visualiser les regroupements

        Étudier la séparation des classes

    ### Étape 4 — Apprentissage semi-supervisé

        Propagation de labels (Label Spreading / Label Propagation)

        Pseudo-labelling sur données non étiquetées

        Entraîner un modèle final sur données enrichies

        Comparer les performances (Accuracy, F1-Score…)

## 📊 Jeu de données
    ### 📌 Composition

        1500 images d’IRM

        1400 images non étiquetées

        100 images annotées (50 normal / 50 cancer)

    ### 📌 Format

        .jpg

        Résolution 512 × 512

        Images en niveaux de gris ou RGB convertible en 1 canal

    ### 📌 Origine

        Jeu fourni dans le cadre du projet pédagogique.

##🔧 Installation
    ### 1. Cloner le repository
        git clone https://github.com/<ton-user>/brain-tumor-semisupervised.git
        cd brain-tumor-semisupervised

### 2. Installer les dépendances
    pip install -r requirements.txt

### 3. Placer les données

    Copier le dossier mri_dataset_brain_cancer_oc/ dans data/.

## 🧪 Exécution
    Lancer les notebooks
    jupyter notebook


    Ensuite ouvrir :

    01_exploration_donnees.ipynb

    02_extraction_features.ipynb

    03_semi_supervised.ipynb

## 📘 Technologies utilisées

    Python 3.10

    PyTorch / Torchvision

    scikit-learn

    NumPy / Pandas

    Matplotlib / Seaborn

    UMAP / t-SNE

## 📈 Résultats attendus

    Visualisation claire du dataset (EDA images)

    Extraction efficace des features via CNN pré-entraîné

    Regroupement cohérent à l’aide de clustering

    Amélioration de la classification grâce au semi-supervisé

    Recommandations pour passer à l’échelle (4M d’images)

## 📝 .gitignore (extrait)
    data/
    *.jpg
    *.png
    __pycache__/
    .ipynb_checkpoints/
    .env

## 🎤 Auteur

    Projet réalisé par Alexandre Ba,
    dans le cadre de la formation Data Scientist Machine Learning.

## 📄 Licence

    Projet à usage pédagogique.