# 📓 Notebooks - Exploration & Modélisation

> Documentation des approches de modélisation pour l'analyse de sentiments

---

## Vue d'ensemble des Notebooks

| Notebook | Description |
|----------|-------------|
| `p7_EDA.ipynb` | Analyse exploratoire des données | 
| `p7_modele_simple.ipynb` | Modèles classiques (LogReg + embeddings) |
| `p7_DL_simple.ipynb` | Deep Learning avec embeddings |
| `p7_DL_avance.ipynb` | LSTM & LSTM Bidirectionnel & Transfer Learning DistilBERT| 
| `test_model_distilBert.ipynb` | Test modele DistilBERT |

---

## Analyse Exploratoire (EDA)

### Dataset Original : Sentiment140

| Caractéristique | Valeur |
|-----------------|--------|
| **Source** | Sentiment140 (Kaggle) |
| **Fichier** | `training.1600000.processed.noemoticon.csv` |
| **Volume total** | 1,600,000 tweets |
| **Encoding** | Latin-1 |
| **Classes** | 0 (négatif), 4 (positif) → remappé en 0/1 |

### Structure des données

| Colonne | Description |
|---------|-------------|
| `target` | Sentiment (0: négatif, 1: positif) |
| `id` | Identifiant du tweet |
| `date` | Date de publication |
| `query` | Requête utilisée (NO_QUERY) |
| `user` | Nom d'utilisateur |
| `text` | Contenu du tweet |

### Insights clés

**Distribution des classes**
- Dataset parfaitement équilibré : 800,000 tweets négatifs / 800,000 tweets positifs
- Remapping effectué : valeur `4` → `1` pour avoir des labels binaires (0/1)

**Sous-échantillonnage**
- Dataset de travail : 200,000 tweets (100K négatifs + 100K positifs)
- Échantillon final utilisé : 100,000 tweets pour l'entraînement

**Analyse des longueurs de texte**
- Statistiques calculées : moyenne, médiane, max, min (caractères et mots)
- Seuil observé : majorité des tweets < 500 caractères

### Comparaison des Tokenizers

Deux approches testées sur des exemples de tweets :

| Tokenizer | Avantages | Inconvénients |
|-----------|-----------|---------------|
| `word_tokenize` (NLTK) | Standard, simple | Perd les @mentions, #hashtags, émoticônes |
| `TweetTokenizer` (NLTK) | Préserve @, #, négations, émoticônes, réduit répétitions | Spécifique Twitter |

**Choix retenu : TweetTokenizer**

```python
from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer(
    preserve_case=False,   # Uniformisation en minuscules
    reduce_len=True,       # "loooove" → "loove"
    strip_handles=False    # Conserve les @mentions
)
```

**Justification** : Le TweetTokenizer est optimisé pour le langage Twitter et préserve des éléments sémantiques importants (mentions, hashtags, émoticônes, négations).

---

## Modèle Simple - Logistic Regression

### Stratégies de Preprocessing testées

Trois approches de nettoyage ont été comparées :

#### Preprocess 1 : Nettoyage agressif

```python
def preprocess_1(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"\@\w+|\#", "", text)                  # Mentions, hashtags
    text = re.sub(r"[^\w\s]", "", text)                   # Ponctuation
    text = re.sub(r"\d+", "", text)                       # Chiffres
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords]    # Stopwords
    tokens = [lemmatizer.lemmatize(w) for w in tokens]    # Lemmatisation
    return " ".join(tokens)
```

| Technique | Appliquée |
|-----------|:---------:|
| Lowercase | ✅ |
| Suppression URLs | ✅ |
| Suppression mentions/hashtags | ✅ |
| Suppression ponctuation | ✅ |
| Suppression chiffres | ✅ |
| Suppression stopwords | ✅ |
| Lemmatisation | ✅ |

#### Preprocess 2 : TweetTokenizer + Lemmatisation

```python
def preprocess_2(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"\d+", "", text)                       # Chiffres
    text = re.sub(r"\@\w+|\#", "", text)                  # Mentions, hashtags
    text = re.sub(r"\b[a-z]\b", "", text)                 # Mots 1 lettre
    tokens = tweetTokenize(text)                          # TweetTokenizer
    tokens = [lemmatizer.lemmatize(w) for w in tokens]    # Lemmatisation
    return " ".join(tokens)
```

| Technique | Appliquée |
|-----------|:---------:|
| Suppression URLs | ✅ |
| Suppression chiffres | ✅ |
| Suppression mentions/hashtags | ✅ |
| Suppression mots 1 lettre | ✅ |
| TweetTokenizer | ✅ |
| Lemmatisation | ✅ |

#### Preprocess 3 : TweetTokenizer + Stemming

```python
def preprocess_3(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"\d+", "", text)                       # Chiffres
    text = re.sub(r"\@\w+|\#", "", text)                  # Mentions, hashtags
    tokens = tweetTokenize(text)                          # TweetTokenizer
    tokens = [stemmer.stem(t) for t in tokens]            # Stemming
    return " ".join(tokens)
```

| Technique | Appliquée |
|-----------|:---------:|
| Suppression URLs | ✅ |
| Suppression chiffres | ✅ |
| Suppression mentions/hashtags | ✅ |
| TweetTokenizer | ✅ |
| PorterStemmer | ✅ |

### Embeddings testés

| Embedding | Dimension | Source | Description |
|-----------|-----------|--------|-------------|
| **TF-IDF** | taille vocabulaire | scikit-learn | Fréquence pondérée des termes |
| **Word2Vec** | 100d | Google News | Embeddings pré-entraînés |
| **GloVe** | 25d/100d  | Stanford | Global Vectors |
| **USE** | 512d | TensorFlow Hub | Universal Sentence Encoder |

### Comparaison des Preprocessing (TF-IDF + LogReg)

| Run Name | Preprocessing | Accuracy | F1-Score | Precision |
|----------|---------------|----------|----------|-----------|
| `LR_TFIDF_preprocess_1_wordTokenize_lemmatization` | word_tokenize + lemmatisation | 0.774 | 0.776 | 0.768 |
| `LR_TFIDF_preprocess_2_tweetTokenize_lemmatization`✅  | TweetTokenizer + lemmatisation | **0.795** | **0.795** | **0.794** |
| `LR_TFIDF_preprocess_3_tweetTokenize_stemming` | TweetTokenizer + stemming | 0.795 | 0.795 | 0.794 |


**Observation** : 
Le **Preprocess 2** est meilleur que **Preprocess 1**  
Le **Preprocess 2** est plus rapide (temps d'exécution) que **Preprocess 3**  
---

### Embeddings testés avec Logistic Regression

| Embedding | Type | Taille |
|-----------|------|--------|
| **TF-IDF** | Sparse | Vocabulaire (max_features) |
| **Word2Vec** | Dense | 300d |
| **GloVe** | Dense | 25d / 100d |
| **USE** | Dense | 512d |

### Résultats par Embedding

#### TF-IDF (meilleur preprocessing)

| Run Name | Accuracy | F1-Score | Precision |
|----------|----------|----------|-----------|
| `LR_TFIDF_preprocess_2_tweetTokenize_lemmatization` | **0.795** | **0.795** | **0.794** |

#### Word2Vec

| Run Name | Accuracy | F1-Score | Precision |
|----------|----------|----------|-----------|
| `LR_Word2Vec_optuna` | 0.736 | 0.734 | 0.741 |

#### GloVe

| Run Name | Dimension | Accuracy | F1-Score | Precision |
|----------|-----------|----------|----------|-----------|
| `LR_glove_25D` | 25d | 0.570 | 0.553 | 0.578 |
| `LR_glove_100D` | 100d | 0.593 | 0.576 | 0.602 |


#### USE (Universal Sentence Encoder)

| Run Name | Accuracy | F1-Score | Precision |
|----------|----------|----------|-----------|
| `LR_USE_GridSearch` | **0.782** | **0.781** | **0.786** |
| `LR_USE_optuna` | 0.778 | 0.777 | 0.775 |

---

### Tableau récapitulatif Logistic Regression

| Embedding | Meilleur Run | Accuracy | F1-Score | Precision |
|-----------|--------------|----------|----------|-----------|
| **TF-IDF** | preprocess_2 + lemmatisation | **0.795** | **0.795** | **0.794** |
| **USE** | GridSearch | 0.782 | 0.781 | 0.786 |
| **Word2Vec** | Optuna | 0.736 | 0.734 | 0.741 |
| **GloVe 100d** | - | 0.593 | 0.576 | 0.602 |

**Conclusions** :
- **TF-IDF** domine avec le bon preprocessing (TweetTokenizer + Lemmatisation)
- **USE** offre de bonnes performances sans preprocessing complexe
- **GloVe** sous-performe significativement sur ce dataset Twitter
- Le choix du **preprocessing** a plus d'impact que le choix de l'embedding pour TF-IDF

---
## Deep Learning Simple

### Preprocessing utilisé : Soft Preprocess

Un preprocessing léger pour conserver le maximum d'information sémantique :

```python
def soft_preprocess(text):
    text = str(text).lower()              # Minuscules
    text = re.sub(r"@\w+", "", text)      # Supprimer mentions
    text = re.sub(r"#(\w+)", r"\1", text) # Conserver hashtag sans #
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

### Configuration commune

| Paramètre | Valeur |
|-----------|--------|
| `MAX_TWEET_LENGTH` | 50 |
| `BATCH_SIZE` | 128 |
| `EPOCHS` | 20 |
| `Optimizer` | AdamW |
| `Loss` | Binary Crossentropy |
| `GPU` | Apple M2 |

### Architecture générale

```
Input (text, dtype=string)
    → TextVectorization (vocabulaire adapté sur train)
    → Embedding Layer (pré-entraîné ou appris)
    → GlobalAveragePooling1D
    → Dense(64, relu) + Dropout(0.3)
    → Dense(32, relu) + Dropout(0.3)
    → Dense(1, sigmoid)
```

### Callbacks utilisés

| Callback | Configuration |
|----------|---------------|
| **EarlyStopping** | `monitor='val_loss'`, `patience=3`, `restore_best_weights=True` |
| **ReduceLROnPlateau** | `monitor='val_loss'`, `factor=0.5`, `patience=2` |

---

### 1 Embedding appris (from scratch)

Embedding entraîné directement sur les données Twitter.

| Paramètre | Valeur |
|-----------|--------|
| `VOCAB_SIZE` | ~33K - 73K  |
| `EMBEDDING_DIM` | 32 |

**Résultats MLflow** : `DL_Simple_softPreProcess`

| Vocab Size | Accuracy | F1-Score | Precision |
|------------|----------|----------|-----------|
| 33,655 | 0.763 | 0.762 | 0.763 |
| 72,914 | 0.748 | 0.750 | 0.741 |

---

### 2 Word2Vec (pré-entraîné sur données)

Embeddings Word2Vec entraînés sur le corpus Twitter avec Gensim.

```python
from gensim.models import Word2Vec

modele_word2vec = Word2Vec(
    sentences=train_tokens,
    vector_size=EMBEDDING_DIM,  # 100 dimensions
    window=WINDOW,              # 10
    min_count=2,
    workers=4,
    sg=1,                       # Skip-gram
    seed=42,
    epochs=EPOCHS,
)
```

| Paramètre | Valeur |
|-----------|--------|
| `VOCAB_SIZE` | ~50K |
| `EMBEDDING_DIM` | 100 |
| `WINDOW` | 10 |
| `Trainable` | True |

**Résultats MLflow** : `DL_Simple_Word2Vec_SoftPreProcess`

| Vocab Size | Accuracy | F1-Score | Precision |
|------------|----------|----------|-----------|
| 50,906 | 0.756 | 0.757 | 0.749 |

---

### 3 GloVe (pré-entraîné Twitter)

Embeddings GloVe pré-entraînés sur Twitter (Stanford).

```python
# Source: glove.twitter.27B.100d
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for i, word in enumerate(vocab):
    if word in glove_embeddings:
        embedding_matrix[i] = glove_embeddings[word]
        words_found += 1
```

| Paramètre | Valeur |
|-----------|--------|
| `VOCAB_SIZE` | ~50K |
| `EMBEDDING_DIM` | 100 |
| `Source` | `glove.twitter.27B.100d` |
| `Trainable` | True |

**Résultats MLflow** : `DL_Simple_GloVe_SoftPreProcess`

| Vocab Size | Accuracy | F1-Score | Precision |
|------------|----------|----------|-----------|
| 50,905 | 0.761 | 0.761 | 0.756 |

---

### 4 USE - Universal Sentence Encoder

Embeddings de phrases complets (Google TensorFlow Hub).

| Paramètre | Valeur |
|-----------|--------|
| `EMBEDDING_DIM` | 512 |
| `Source` | TensorFlow Hub |
| `Trainable` | False (frozen) |

**Résultats MLflow** : `DL_USE_Dense_softPreProcess`

| Run | Accuracy | F1-Score | Precision |
|-----|----------|----------|-----------|
| Run 1 | 0.774 | 0.782 | 0.752 |
| Run 2 (best) | 0.766 | 0.773 | 0.748 |

---

### Tableau récapitulatif DL Simple

| Modèle | Embedding | Dim | Accuracy | F1-Score | Precision |
|--------|-----------|-----|----------|----------|-----------|
| DL Simple | Appris (scratch) | 32 | 0.763 | 0.762 | 0.763 |
| DL Simple | Word2Vec | 100 | 0.756 | 0.757 | 0.749 |
| DL Simple | GloVe Twitter | 100 | 0.761 | 0.761 | 0.756 |
| DL Simple | USE | 512 | **0.774** | **0.782** | 0.752 |

**Observation** : USE (Universal Sentence Encoder) obtient les meilleures performances grâce à ses embeddings de phrases pré-entraînés sur un large corpus.

---
## Deep Learning Avancé

### Preprocessing utilisé : Soft Preprocess

Même preprocessing léger que pour DL Simple :

```python
def soft_preprocess(text):
    text = str(text).lower()              # Minuscules
    text = re.sub(r"@\w+", "", text)      # Supprimer mentions
    text = re.sub(r"#(\w+)", r"\1", text) # Conserver hashtag sans #
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

---

### LSTM Simple

#### Architecture

```
Input (text, dtype=string)
    → TextVectorization
    → Embedding (EMBEDDING_DIM, mask_zero=True)
    → LSTM(64, dropout=0.2)
    → Dense(64, relu) + Dropout(0.4)
    → Dense(32, relu) + Dropout(0.3)
    → Dense(1, sigmoid)
```

#### Configuration

| Paramètre | Valeur |
|-----------|--------|
| `VOCAB_SIZE` | ~32K (adapté) |
| `MAX_TWEET_LENGTH` | 50 |
| `EMBEDDING_DIM` | 32 |
| `LSTM units` | 64 |
| `Dropout LSTM` | 0.2 |
| `Optimizer` | AdamW (lr=0.001) |
| `Callbacks` | EarlyStopping (patience=5), ReduceLROnPlateau |

#### Résultats MLflow : `DL_LSTM_softPreProcess`

| Run | Accuracy | F1-Score | Precision | Temps |
|-----|----------|----------|-----------|-------|
| Run 1 | 0.749 | 0.750 | 0.743 | 30.6 min |
| Run 2 | 0.751 | 0.737 | 0.776 | 23.6 min |
| Run 3 | 0.755 | 0.755 | 0.751 | 24.8 min |

---

### LSTM Bidirectionnel

#### Architecture

```
Input (text, dtype=string)
    → TextVectorization
    → Embedding (EMBEDDING_DIM, mask_zero=True)
    → SpatialDropout1D(0.2)
    → Bidirectional(LSTM(64, dropout=0.2))  # Output: 128 dim
    → Dense(64, relu) + Dropout(0.4)
    → Dense(32, relu) + Dropout(0.3)
    → Dense(1, sigmoid)
```

#### Configuration

| Paramètre | Valeur |
|-----------|--------|
| `VOCAB_SIZE` | ~32K (adapté) |
| `EMBEDDING_DIM` | 32 |
| `LSTM units` | 64 (x2 = 128 avec Bidirectional) |
| `SpatialDropout1D` | 0.2 |
| `Optimizer` | AdamW (lr=5e-4) |

#### Résultats MLflow : `DL_LSTM_Bidirectionnel_softPreProcess`

| Run | Accuracy | F1-Score | Precision | Temps |
|-----|----------|----------|-----------|-------|
| Run 1 | 0.754 | 0.743 | 0.776 | 4.7h |
| Run 2 | 0.689 | 0.658 | 0.726 | 3.7h |

---

### DistilBERT (Transfer Learning)

#### Architecture

```
Inputs:
    → input_ids (MAX_LENGTH,)
    → attention_mask (MAX_LENGTH,)
    
DistilBERT (distilbert-base-uncased)
    → CLS token extraction (last_hidden_state[:, 0, :])
    → Dropout(0.2)
    → Dense(128, relu)
    → Dropout(0.1)
    → Dense(1, sigmoid)
```

#### Configuration

| Paramètre | Valeur |
|-----------|--------|
| `MODEL_NAME` | `distilbert-base-uncased` |
| `MAX_LENGTH` | 128 |
| `BATCH_SIZE` | 16 |
| `Learning Rate` | Variable |
| `Optimizer` | AdamW |

#### Tokenisation DistilBERT

```python
from transformers import AutoTokenizer, TFAutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")

def tokenize_texts(texts, tokenizer, max_length):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf",
    )
```

#### Résultats MLflow : `DL_DistilBERT`

| Run | Trainable | Epochs | Accuracy | F1-Score | Precision | Temps |
|-----|-----------|--------|----------|----------|-----------|-------|
| `trainable_False_128` | ❌ | 3 | 0.747 | 0.762 | 0.715 | 4.4h |
| `trainable_False_EPOCHS_5_128` | ❌ | 5 | 0.745 | 0.744 | 0.745 | 56.6 min |
| `trainable_False_EPOCHS_20_128` | ❌ | 20 | 0.765 | 0.772 | 0.746 | 4.2h |
| `trainable_False_128` | ❌ | 3 | 0.748 | 0.731 | 0.780 | 38 min |
| `trainable_128_test_sample1000` | ✅ | 1 | 0.749 | 0.743 | 0.797 | 6.8 min |
| `trainable_False_EPOCHS_5_128` | ✅ | 5 | **0.814** | **0.810** | **0.827** | 4.0 |
| `trainable_EPOCHS_3_128_best` | ✅ | 3 | **0.809** | **0.795** | **0.854** | 5.0d |

**Meilleur modèle** : `DL_DistilBERT_trainable_EPOCHS_3_128_best` avec fine-tuning complet.

---

### Tableau récapitulatif DL Avancé

| Architecture | Meilleur Run | Accuracy | F1-Score | Precision | 
|--------------|--------------|----------|----------|-----------|
| LSTM Simple | softPreProcess_32890 | 0.755 | 0.755 | 0.751 | 
| LSTM Bidirectionnel | softPreProcess_32890 | 0.754 | 0.743 | 0.776 | 
| **DistilBERT (frozen)** | trainable_False_EPOCHS_20 | 0.765 | 0.772 | 0.746 | 
| **DistilBERT (fine-tuned)** | trainable_EPOCHS_3_best | **0.809** | **0.795** | **0.854** |

**Observations** :
- **DistilBERT fine-tuné** atteint les meilleures performances (~81% accuracy)
- Le **fine-tuning** (trainable=True) améliore significativement les résultats (+6% accuracy)
- **Temps d'entraînement** très long pour DistilBERT fine-tuné 


---

###  Contraintes de déploiement DistilBERT

**Conclusion** : Malgré d'excellentes performances, DistilBERT ne peut pas être déployé sur Heroku gratuit en raison des contraintes mémoire. (utilisation de Tensorflow lite)

---

## Tableau Récapitulatif Global

| Approche | Modèle | Embedding/Config | Accuracy | F1-Score | Precision | 
|----------|--------|------------------|----------|----------|-----------|
| **Simple** | LogReg | TF-IDF + Preprocess_2 | **0.795** | **0.795** | 0.794 | 
| Simple | LogReg | USE | 0.782 | 0.781 | 0.786 | 
| Simple | LogReg | Word2Vec | 0.736 | 0.734 | 0.741 |
| Simple | LogReg | GloVe 100d | 0.593 | 0.576 | 0.602 | 
| DL Simple | Dense + GAP | Embedding appris | 0.763 | 0.762 | 0.763 | 
| DL Simple | Dense + GAP | Word2Vec | 0.756 | 0.757 | 0.749 | 
| DL Simple | Dense + GAP | GloVe | 0.761 | 0.761 | 0.756 | 
| DL Simple | Dense + GAP | USE | 0.774 | 0.782 | 0.752 | 
| DL Avancé | LSTM | Embedding appris | 0.755 | 0.755 | 0.751 | 
| DL Avancé | BiLSTM | Embedding appris | 0.754 | 0.743 | 0.776 | 
| DL Avancé | DistilBERT (frozen) | - | 0.765 | 0.772 | 0.746 | 
| **DL Avancé** | **DistilBERT (fine-tuned)** | - | **0.809** | 0.795 | **0.854** | 

---

## Choix du Modèle Final

### Analyse des candidats

| Modèle | Accuracy | Taille | Temps inférence | Verdict |
|--------|----------|--------|-----------------|---------|
| DistilBERT| **0.809** | ~250 MB + deps | ~100-500ms | ❌ Mémoire Heroku |
| TF-IDF + LogReg | 0.795 | ~10-50 MB | < 10ms | ✅ **Retenu** |

=> Après une réduction du modele DistilBERT via TFLite à ~90% ensuite à ~67%, le modèle a subit une dégradation sévère(modèle inexploitable)  

### Modèle retenu : TF-IDF + Logistic Regression

| Avantage | Détail |
|----------|--------|
| ✅ **Performance solide** | 79.5% accuracy, 79.5% F1-score |
| ✅ **Très léger** | ~10-50 MB (vectorizer + modèle .joblib) |
| ✅ **Ultra rapide** | Inférence < 10ms |
| ✅ **Simple** | Pas de GPU, pas de dépendances lourdes |
| ✅ **Maintenance facile** | scikit-learn uniquement |

### Écart avec DistilBERT

| Métrique | TF-IDF + LogReg | DistilBERT | Écart |
|----------|-----------------|------------|-------|
| Accuracy | 0.795 | 0.809 | -1.4% |
| F1-Score | 0.795 | 0.795 | 0% |
| Precision | 0.794 | 0.854 | -6% |

**Conclusion** : L'écart de performance (~1.4% accuracy) ne justifie pas la complexité et les contraintes de déploiement de DistilBERT pour ce prototype.

---

## Suivi MLflow

### Configuration

```python
import mlflow

mlflow.set_experiment("p7_air_paradis")

with mlflow.start_run(run_name="tfidf_logreg_v1"):
    mlflow.log_params({...})
    mlflow.log_metrics({...})
    mlflow.sklearn.log_model(model, "model")
```

### Métriques trackées

- `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
- `train_*` vs `test_*` (détection overfitting)
- Hyperparamètres : `C`, `solver`, `max_iter`
- Temps d'entraînement

### Optimisation Hyperparamètres

- **Outil** : Optuna
- **Stratégie** : TPE Sampler + MedianPruner
- **Paramètres optimisés** : `C`, `solver`

### Accès à l'interface MLflow

```bash
cd notebook
mlflow ui --port 5000
# Ouvrir http://localhost:5000
```

---

## Description des fichiers

| Fichier | Contenu |
|---------|---------|
| `p7_EDA.ipynb` | Chargement données, statistiques, visualisations, comparaison tokenizers |
| `p7_modele_simple.ipynb` | LogReg avec TF-IDF, Word2Vec, GloVe, USE + optimisation Optuna |
| `p7_DL_simple.ipynb` | Réseaux denses avec embeddings pré-entraînés |
| `p7_DL_avance.ipynb` | LSTM simple et bidirectionnel + DistilBERT|
| `test_model_distilBert.ipynb` | Evaluation Distilbert |

---
