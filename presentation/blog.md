# Analyse de sentiment Twitter : Construction d'un système de prédiction avec une approche MLOps

## Problématique

Air Paradis, compagnie aérienne, a sollicité notre cabinet MIC pour développer un outil capable de prédire le sentiment lié à un tweet.  
**L'objectif : anticiper les bad buzz avant qu'ils ne deviennent viraux.**

Ce blog revient sur les différentes étapes du projet, de l'expérimentation des modèles jusqu'au déploiement en production.

![Structure du projet](assets/structure_projet.png)  
*Architecture du projet sur GitHub*

---

## Sommaire

1. [Données et exploration](#données-et-exploration)
2. [Expérimentation et modélisation](#expérimentation-et-modélisation)
3. [Le défi TensorFlow Lite](#le-défi-tensorflow-lite)
4. [Architecture MLOps](#architecture-mlops)
5. [Monitoring et alerting](#monitoring-et-alerting)
6. [Déploiement et production](#déploiement-et-production)
7. [Résultats finaux](#résultats-finaux)

---

## Données et exploration

La première étape consiste à explorer le dataset pour comprendre la nature des données.

**Dataset** : [Sentiment140 (Kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140)  
- 1,6 million de tweets répartis équitablement (positif / négatif)  
- Échantillon retenu : **50 000 tweets**

**Variables clés** :
- `target` : polarité (0 = négatif, 1 = positif)
- `text` : contenu du tweet
- `user`, `date`, `id` : métadonnées

**Insights :**
- Distribution équilibrée des classes (50 / 50)  
- Longueur moyenne des tweets : 80–120 caractères  
- Présence de bruit (URLs, mentions, hashtags, dédoublement de lettres…)

---

## Expérimentation et modélisation

Nous avons testé trois approches pour identifier le modèle optimal.  
Le choix final s'appuie sur la **précision**, métrique critique dans notre contexte, où les **faux positifs** (tweets négatifs prédits comme positifs) sont particulièrement problématiques pour la détection de bad buzz.

### Approche 1 : Modèles sur mesure simples

Trois stratégies de preprocessing testées :

1. **word_tokenize + lemmatisation**  
2. **TweetTokenizer + lemmatisation**  
3. **TweetTokenizer + stemming**

**Vectorisation** : TF-IDF (comptage pondéré des mots)

![Comparaison preprocessing](assets/compare_preprocess_LR.png)

**Embeddings testés** : 
- Word2Vec (pré-entraîné)  
- GloVe (pré-entraîné)  
- Universal Sentence Encoder (USE)

> **Note :** Nous avons utilisé d’abord **GridSearchCV** pour explorer une petite grille d’hyperparamètres, puis **Optuna** pour élargir l’espace de recherche de manière plus efficace et adaptative.

**Insight principal :** le preprocessing a un impact plus significatif sur les performances que le choix de l'embedding lui-même.

![Comparaison modèle sur mesure simple](assets/compare_modele_LR.png)

**Configuration commune à tous les modèles Deep Learning :**

Pour la suite des expérimentations, nous avons exploité le **GPU Apple M2** via TensorFlow pour macOS et l'extension `tensorflow-metal` afin d'accélérer l'entraînement.  
L'utilisation de `tf.data.Dataset` permet d'optimiser le pipeline de données, notamment grâce au **prefetching**, qui parallélise le chargement des batchs avec l'exécution du modèle.

![Fonction make_dataset](assets/make_dataset.png)

- **Optimiseur** : AdamW (Adam avec weight decay pour meilleure régularisation)
- **Callbacks** :
  - `EarlyStopping` : arrêt anticipé si validation loss stagne (patience=5)
  - `ReduceLROnPlateau` : réduction du learning rate (factor=0.5, patience=3)
- **Dataset** : 50 000 tweets (70% train / 15% val / 15% test)

---

### Approche 2 : Approche sur mesure avancée

#### Modèle 1 : Deep Learning simple 

Nous avons testé **4 stratégies d'embeddings** pour évaluer l'impact de la représentation vectorielle sur les performances :


##### 1.1 Embedding entraîné sur le corpus

**Architecture** :
![Architecture DL Simple](assets/architecture_DL_Simple.png)

##### 1.2 Word2Vec (pré-entraîné)
**Architecture** :
- Chargement des embeddings Word2Vec (Google News, 100 dimensions)
- `Embedding` initialisé avec weights Word2Vec (trainable=True)
![Architecture DL Simple Word2vec](assets/architecture_DL_Word2vec.png)

##### 1.3 GloVe (pré-entraîné)

**Architecture** :
- Chargement des embeddings GloVe (Twitter, 100 dimensions)
- `Embedding` initialisé avec weights GloVe (trainable=True)
![Architecture DL Simple GloVe](assets/architecture_DL_gloVe.png)

##### 1.4 Universal Sentence Encoder (USE)
- Chargement des embeddings USE (512 dimensions)
![Architecture DL Simple USE](assets/architecture_DL_USE.png)

---

#### Modèle 2 : LSTM (Long Short-Term Memory)
Les LSTM permettent de capturer les **dépendances temporelles** dans les séquences de texte grâce à leur mécanisme de mémoire à long terme.
##### 2.1 LSTM  
![Architecture DL LSTM](assets/architecture_LSTM.png)

##### 2.2 LSTM Bidirectionnel
![Architecture DL BiLSTM](assets/architecture_BiLSTM.png)

**Comparaison des stratégies d'embedding + LSTM + BiLSTM :**
![Comparaison approche 2](assets/comparaison_DL_SIMPLE_LSTM_BILSTM.png)