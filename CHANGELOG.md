# CHANGELOG - Projet 7 : Analyse de Sentiments Twitter

Documentation des versions principales du projet.

---

## [1.0.0] - 2024-11-17 - VERSION PRODUCTION 

### Modèle : TF-IDF + Logistic Regression

**Fichiers principaux :**
- `app/Model_LR.py` : Nouveau modèle de production
- `app/models_lr/` : Modèle et vectorizer 

**Performance :**
- Accuracy : 0.79
- RAM : ~200 MB
- Déployable sur Heroku 

**Justification :**
- Respecte les contraintes mémoire (512 MB Heroku)
- Performances acceptables
- Stable en production

**Organisation :**
- Notebooks déplacés dans `notebook/`
- Fichiers temporaires archivés dans `dev/`

## [0.3.0] - 2024-11-12 - Tentative  sans TFLite 

### Modèle : DistilBERT 

**Fichiers :**
- `app/Model.py` : DistilBERT 

**Résultat :**
- Accuracy : 0.81
- Consommation RAM: >700MB
- Dépassement limite Heroku (512MB)
- Approche abandonnée contraintes de ressources 

## [0.2.0] - 2024-11-12 - Tentative TFLite 

### Modèle : DistilBERT + TensorFlow Lite

**Fichiers :**
- `app/Model.py` : DistilBERT converti en TFLite
- `app/model.tflite` : Modèle compressé

**Résultat :**
- Dégradation sevère des performances
- Approche abandonnée

**Fichier conservé pour référence et analyse future**


## [0.1.0] - 2024-11-13 - Exploration 

### Phase d'exploration et expérimentation

**Travail effectué :**
- Analyse exploratoire des données (EDA)
- Tests de différentes approches
- Notebooks d'expérimentation

**Notebooks :**
- `notebook/p7_EDA.ipynb`
- `notebook/p7_DL_simple.ipynb`
- `notebook/p7_DL_avance.ipynb`
- `notebook/p7_modele_simple.ipynb`
- `notebook/test_final_model.ipynb`


## Notes

- DistilBERT "pur" (sans TFLite) testé uniquement en local : ~0.82 acc mais >700 MB RAM
- Date de création : 25 novembre 2024

