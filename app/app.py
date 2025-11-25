import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import uvicorn
from fastapi import FastAPI, HTTPException
from .Tweets import Tweet, PredictedResult, UserFeedback
# from .Model import model, tokenizer, predict_sentiment
from .monitoring import log_error_pred, log_prediction
from .Model_LR import predict_sentiment_lr, tfidf, logreg
import time 


# creer l'application FastAPI
app = FastAPI(
    title="API Analyse de Sentiment des Tweets",
    description="Analyser le sentiment des tweets",
    version="1.0.0",
)

Environment = "heroku"

# Index route open automatiquement sur http://
@app.get('/')
def index():
    return {"message": "Analyse de sentiment des tweets",
            "endpoints": {
                "/health": " GET - Vérification de l'état du modèle",
                "/predict": " POST - Prédire le sentiment d'un tweet",
                "/feedback": " POST - Signaler une erreur de prédiction"
            }} 

# # Route de vérification de l'état du modèle distilbert
# # @app.get('/health')
# # def health_check():
# #     if model is None or tokenizer is None:
# #         raise HTTPException(503, "Modèle non disponible")
# #     return {"status": "ok", "model": "loaded", "environment": Environment}

@app.get("/health")
def health_check():
    if tfidf is None or logreg is None:
        raise HTTPException(503, "Modèle LR non disponible")
    return {"status": "ok", "model": "logreg_tfidf"}

# # Route pour les prédictions
# # @app.post('/predict', response_model=PredictedResult)
# # def predict(tweet: Tweet) -> PredictedResult: 
    
# #     # Démarrer le chronomètre
# #     start_time = time.time()
    
# #     # Vérifier que le modèle est chargé
# #     if model is None or tokenizer is None:
# #         raise HTTPException(503, "Modèle non chargé")
    
# #     try:
# #         # Prédiction
# #         result = predict_sentiment(tweet.text)
        
# #         # Calculer le temps de réponse en millisecondes
# #         response_time = (time.time() - start_time) * 1000
        
# #         # Log la prédiction
# #         log_prediction(
# #             text=result.text,
# #             sentiment=result.sentiment,
# #             probability=result.probability,
# #             confidence=result.confidence,
# #             response_time=response_time,
# #             environment=Environment
# #         )
# #         return result
    
# #     except Exception as e:
# #         raise HTTPException(500, f"Erreur de prédiction: {str(e)}")

# Route pour les prédictions avec modèle léger TF-IDF + LogisticRegression
@app.post("/predict", response_model=PredictedResult)
def predict(tweet: Tweet) -> PredictedResult:
    
    # Démarrer le chronomètre
    start_time = time.time()
    
    try:
        # Prédiction
        result = predict_sentiment_lr(tweet.text)   
        
        # Temps de réponse en millisecondes
        response_time = (time.time() - start_time) * 1000
            
        # Log la prédiction
        log_prediction(
            text=result.text,
            sentiment=result.sentiment,
            probability=result.probability,
            confidence=result.confidence,
            response_time=response_time,
            environment=Environment
    )   
        return result
    except Exception as e:
        raise HTTPException(500, f"Erreur de prédiction: {str(e)}")


    
@app.post('/feedback')
def report_error(feedback: UserFeedback):
    try:
        log_error_pred(
            text=feedback.text,
            predicted_sentiment=feedback.predicted_sentiment,
            correct_sentiment=feedback.correct_sentiment,
            confidence=feedback.confidence,
            environment="local"
        )
        return {"status": "success", "message": "Error reported"}
    except Exception as e:
        raise HTTPException(500, f"Erreur lors de l'enregistrement: {str(e)}")




if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)



