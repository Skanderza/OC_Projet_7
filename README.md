# Air Paradis - Analyse de Sentiments Twitter

Prototype IA pour anticiper le bad buzz sur les réseaux sociaux(Twitter)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()

## Contexte du projet

### Le Client

**Air Paradis** une compagnie aérienne (fictive) qui fait face à des défis d'e-réputation sur les réseaux sociaux. Les bad buzz peuvent avoir un impact significatif sur l'image de marque et la confiance des clients.

### La Mission

Le cabinet **MIC (Marketing Intelligence Consulting)** a été missionné pour développer un prototype IA capable de :
- Prédire le sentiment(positif/négatif) associé à un tweet
- Anticiper les bad buzz avant qu'ils ne deviennent viraux 
- Systeme d'alerte si 3  mauvaises predictions en moins de 5 minutes
- Fournir un outil accessible via une interface web simple

### Contraintes

- Coût de mise en production (solution Cloud gratuite Heroku)

## Objectifs

### 1. Modélisation
Comparer plusieurs approches de Machine Learning et Deep Learning :
- **Modèle simple** : Logistic Regression (TF-IDF/Word2vec/GloVe/USE)
- **Modèle Deep learning simple** : Couche embedding + (Word2vec/GloVe/USE)
- **Modèle Deep learning avancé** : LSTM/LSTM bidirectionnel/Distilbert(Transfert learning)

### 2. MLOps
Mettre en œuvre une démarche MLOps complète :
- **Tracking** des expérimentations avec MLflow
- **Pipeline CI/CD** avec GitHub Actions
- **Monitoring** en production avec Azure Application Insights
- **Alertes** automatiques en cas de mauvaises predictions

### 3. Déploiement
- **API REST** de prédiction (FastAPI)
- **Interface utilisateur** (Streamlit)
- **Feedback loop** pour l'amélioration continue

---

## Architecture Globale

L'architecture suit un flux complet de bout en bout :

1. **Interface utilisateur** (Streamlit local) → Saisie du tweet
2. **API REST** (FastAPI sur Heroku) → Traitement et prédiction
3. **Modèle ML** (TF-IDF + LogReg) → Classification du sentiment
4. **Monitoring** (Azure Application Insights) → Suivi des performances et alertes

---

## Données

| Caractéristique | Détail |
|-----------------|--------|
| **Source** | Sentiment140 Dataset |
| **Volume** | 1.6 million de tweets |
| **Volume traité** | 100k de tweets |
| **Format** | Label binaire (0: négatif, 1: positif) |
| **Téléchargement** | [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) |

---
## Structure du Repository

```
OC_Projet_7/
├── .github/
│   └── workflows/          # CI/CD GitHub Actions
├── app/
│   ├── models_lr/          # Modèles Logistic Regression
│   │   ├── logreg_tfidf_133k.joblib
│   │   └── tfidf_vectorizer_133k.joblib
│   ├── .env.example        # Template variables d'environnement
│   ├── Model.py            # Classe modèle de prédiction
│   ├── Model_LR.py         # Modèle Logistic Regression
│   ├── Tweets.py           # Schémas Pydantic
│   ├── app.py              # Application FastAPI
│   ├── monitoring.py       # Intégration Azure Insights
│   └── streamlit_app.py    # Interface utilisateur
│   └── logo.webp    # Logo app
├── models_final/           # Modèles finaux versionnés
│   ├── logreg_tfidf_133k.joblib
│   └── tfidf_vectorizer_133k.joblib
├── notebook/
│   ├── p7_EDA.ipynb                # Analyse exploratoire
│   ├── p7_modele_simple.ipynb      # Regression logistique
│   ├── p7_DL_simple.ipynb          # Deep Learning simple
│   ├── p7_DL_avance.ipynb          # Deep Learning avancé
│   └── test_model_distilBert.ipynb # Expérimentation BERT
├── tests/
│   ├── __init__.py
│   ├── test_app.py             # Tests API
│   ├── test_model_lr.py        # Tests modèle LogReg
│   └── test_model_distilbert.py # Tests modèle BERT
├── presentation/
│   └──BLOG.md                 # Article blog MLOps
├── .dockerignore
├── .gitignore
├── CHANGELOG.md            # Historique des versions
├── Dockerfile              # Configuration Docker
├── Procfile                # Configuration Heroku
├── README.md               # Ce fichier
├── model_tf_lite.py        # Script conversion TF Lite
├── requirements.txt        # Dépendances Python
└── runtime.txt             # Version Python Heroku
```
---

## Quick Start

### Prérequis

- Python 3.11+
- Conda (recommandé) ou pip
- Git

### Installation

```bash
# Cloner le repository
git clone https://github.com/Skanderza/OC_Projet_7.git
cd OC_Projet_7

# Créer l'environnement conda
conda create -n p7_sentiment python=3.11
conda activate p7_sentiment

# Installer les dépendances
pip install -r requirements.txt
```

### Lancement Local

```bash
# Lancer l'API
uvicorn app.app:app --reload --port 8000

# Dans un autre terminal, lancer Streamlit
streamlit run app/streamlit/app.py
```

### Accès

- **Interface Streamlit** : http://localhost:8501
- **API locale** : http://localhost:8000/docs
- **MLflow UI** : http://localhost:5000
- **Heroku** : https://sentiment-twitter-p7-357ab866923c.herokuapp.com/docs (Dyno deconnecté)

---

## Résultats Clés

| Modèle | Accuracy | F1-Score | Precision | Déployé |
|--------|----------|----------|---------|:-------:|
| TF-IDF + Logistic Regression | 0.795 | 0.795 | 0.794 | ✅ |
| DistilBERT | 0.809 | 0.795| 0.854 | ❌ |


**Choix du modèle de production** : TF-IDF + Logistic Regression
- Performances solides
- Compatible avec les contraintes mémoire Heroku (512 MB)
- Temps d'inférence rapide

→ Détails complets dans [notebook/README.md](notebook/README.md)

---

## Documentation

| Document | Description |
|----------|-------------|
| [notebook/README.md](notebook/README.md) | Approches de modélisation et résultats détaillés |
| [app/README.md](app/README.md) | Documentation API et Interface utilisateur |
| [BLOG.md](BLOG.md) | Article blog sur la démarche MLOps |
| [CHANGELOG.md](CHANGELOG.md) | Historique des versions |

---

## 👤 Auteur

**Skander ZAHI**

Ce projet est réalisé dans le cadre de la formation **OpenClassrooms - Parcours Data Scientist**.

Dataset disponible publiquement sur Kaggle.

---