# Application - API & Interface

> API de prédiction de sentiment et interface utilisateur Streamlit

---

## Architecture

```
User (Tweet) 
    → Streamlit (Interface locale)
    → FastAPI (API Heroku)  
    → TF-IDF + LogReg (Modèle)
    → Azure Application Insights (Monitoring)
```

---

## Structure du dossier

```
app/
├── models_lr/                    # Modèles sauvegardés
│   ├── logreg_tfidf_133k.joblib  # Logistic Regression
│   └── tfidf_vectorizer_133k.joblib
├── app.py                        # API FastAPI
├── streamlit_app.py              # Interface utilisateur
├── Model_LR.py                   # Classe de prédiction (TF-IDF + LogReg)
├── Model.py                      # Classe DistilBERT (non utilisé en prod)
├── Tweets.py                     # Schémas Pydantic
├── monitoring.py                 # Intégration Azure Application Insights
├── .env.example                  # Template variables d'environnement
└── logo.webp                     # Logo Air Paradis
```

---

## API FastAPI

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Index - liste des endpoints |
| `GET` | `/health` | Vérification état du modèle |
| `POST` | `/predict` | Prédiction de sentiment |
| `POST` | `/feedback` | Signalement erreur de prédiction |

### Exemple de requête `/predict`

```bash
curl -X POST "https://sentiment-twitter-p7-357ab866923c.herokuapp.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "i love this movie"}'
```

### Réponse

```json
{
  "text": "I love this movie",
  "sentiment": "positive",
  "probability": 0.9386,
  "confidence": 0.9386
}
```

### Documentation Swagger

Accessible à : [/docs](https://sentiment-twitter-p7-357ab866923c.herokuapp.com/docs)

---

## Interface Streamlit

### Fonctionnalités

- Saisie de tweet (max 128 caractères)
- Affichage du sentiment prédit (positif/négatif)
- Probabilité et niveau de confiance
- **Feedback utilisateur** : thumbs up/down pour signaler les erreurs

### Lancement local

```bash
# Variable d'environnement
export RUN_MODE=local

# Lancer Streamlit
streamlit run app/streamlit_app.py
```

---

## Monitoring Azure

### Événements trackés

| Type | Description |
|------|-------------|
| `log_prediction` | Chaque prédiction (texte, sentiment, confiance, temps) |
| `log_error_pred` | Misprediction signalée par l'utilisateur |
| `log_api_health` | État de santé de l'API |

### Alerte configurée

> **Trigger** : 3 mauvaises predictions en moins 5 minutes → Email notification zahiskander@gmail.com

---

## Configuration

### Variables d'environnement

```bash
# .env
APPINSIGHTS_INSTRUMENTATION_KEY=your_key_here
RUN_MODE=local  # ou "prod"
```

### Déploiement Heroku

```bash
# Procfile
web: uvicorn app.app:app --host 0.0.0.0 --port $PORT
```

---

## Liens

| Ressource | URL |
|-----------|-----|
| **API Production** | [https://sentiment-twitter-p7-357ab866923c.herokuapp.com/docs](https://sentiment-twitter-p7-357ab866923c.herokuapp.com/docs) |
| **Streamlit** (local)| http://localhost:8501/|