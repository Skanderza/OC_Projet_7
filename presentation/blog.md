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
