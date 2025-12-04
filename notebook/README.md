# Notebooks - Exploration & Modélisation

> Documentation des approches de modélisation pour l'analyse de sentiments Twitter

---

## Vue d'ensemble des Notebooks

| Notebook |Description | Contenu |
|---------|---------|---------|
| `p7_EDA.ipynb` | EDA| Analyse exploratoire, comparaison tokenizers |
| `p7_modele_simple.ipynb` | Modèle sur mesure simple| LogReg + TF-IDF/Word2Vec/GloVe/USE |
| `p7_DL_simple.ipynb` |Modèle sur mesure avancé | Deep Learning + Embeddings_corpus/Word2Vec/GloVe/USE|
| `p7_DL_avance.ipynb` | Modèle sur mesure avancé| LSTM, LSTM-Bidirectionnel, DistilBERT |

---

## Analyse Exploratoire (EDA)

| Caractéristique | Valeur |
|-----------------|--------|
| **Source** | Sentiment140 (Kaggle) |
| **Volume total** | 1.6M tweets |
| **Échantillon utilisé** | 50K tweets |
| **Classes** | Binaire (0: négatif, 1: positif) |
| **Distribution** | Équilibrée (50/50) |

**Insight clé** : Le `TweetTokenizer` (NLTK) surpasse le `word_tokenize` standard en préservant les éléments sémantiques propres à Twitter (mentions, hashtags, émoticônes, négations).

---

## Résultats des Modèles

### Modèles Candidats au Déploiement

| Modèle | Accuracy | F1-Score | Precision | Déployable Heroku |
|--------|----------|----------|-----------|:-----------------:|
| **TF-IDF + Logistic Regression** | 0.795 | 0.795 | 0.794 | ✅ |
| **DistilBERT (fine-tuned )** | 0.809 | 0.795 | 0.854 | ❌ |
| **DistilBERT ( réduction TFLite)** | 50% | 50% | 50% | ❌ |

---

### Autres Modèles Explorés

#### Logistic Regression + Embeddings

| Embedding | Accuracy | F1-Score |Precision |
|-----------|----------|----------|----------|
| TF-IDF | **0.795** | **0.795** |**0.794** |
| USE | 0.782 | 0.781 |0.786|
| Word2Vec | 0.736 | 0.734 |0.741|
| GloVe 100d | 0.593 | 0.576 | 0.602|

#### Deep Learning Simple (Dense + GAP)

| Embedding | Accuracy | F1-Score |Precision |
|-----------|----------|----------|----------|
| USE | **0.766** | **0.773** |0.748 |
| Embedding corpus | 0.763 | 0.762 |**0.763** |
| GloVe | 0.761 | 0.761 |0.756 |
| Word2Vec | 0.756 | 0.757 |0.749 |

#### Deep Learning Avancé

| Architecture | Accuracy | F1-Score |Precision |
|--------------|----------|----------|----------|
| DistilBERT (fine-tuned) | 0.809 | 0.795|**0.854** |
| DistilBERT (frozen) | **0.814** | **0.809** |0.827 |
| LSTM | 0.751 | 0.737 |0.776 |
| BiLSTM | 0.761 | 0.761 |0.756 |

---

## Choix du Modèle Final

| Critère | TF-IDF + LogReg | DistilBERT |
|---------|-----------------|------------|
| Accuracy | 0.795 | 0.809 |
| Taille | 2 MB | 766 MB |
| Inférence |  ~15ms | ~100-500ms |
| Heroku  | ✅ | ❌ |

**Modèle retenu** : **TF-IDF + Logistic Regression**

L'écart de 1.4% d'accuracy ne justifie pas les contraintes de déploiement de DistilBERT(766 MB vs 2 MB)
qui est de tte façcon impossible sur Heroku (512 MB).

> Note : La conversion TFLite de DistilBERT a été tentée mais a dégradé sévèrement les performances du modèle.

---

## Suivi des Expérimentations

### MLflow
Toutes les expérimentations sont trackées avec MLflow :
- Métriques : accuracy, F1, precision, recall, ROC-AUC
- Artefacts : modèles, courbes ROC
- Model Registry pour le versioning

```bash
mlflow ui --port 5000
```

### Optuna
Recherche d'hyperparamètres avec Optuna pour le modèle surmesure simple:
- **Stratégie** : TPE Sampler + MedianPruner
- **Paramètres optimisés** : `C`, `solver` , `penalty`, `batch_size`, `max_iter`

---

