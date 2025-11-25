from opencensus.ext.azure.log_exporter import (
    AzureLogHandler,
)  # Pour envoyer des logs vers Azure

import logging  # Pour créer des logs (messages)
import os
from dotenv import load_dotenv
from datetime import datetime

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
INSTRUMENTATION_KEY = os.getenv("APPINSIGHTS_INSTRUMENTATION_KEY")

# Configurer le logging pour envoyer les logs vers Azure Application Insights
def setup_logging():
    if not INSTRUMENTATION_KEY:
        raise ValueError(
            "APPINSIGHTS_INSTRUMENTATION_KEY not found in environment variables"
        )

    # Configurer le logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    print(f"=>{__name__}")
    # Handler 1 => Azure
    azure_handler = AzureLogHandler(
        connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"
    )
    logger.addHandler(azure_handler)
    
    # Handler 2 => local
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()


# Enregistrer une prédiction dans les logs Azure
def log_prediction(text: str, sentiment: str, probability: float, confidence: float, response_time: float, environment="local"):
    logger.info(
        "New prediction made",
        extra={
            "custom_dimensions": {
                "environment": environment,
                "text": text,
                "sentiment": sentiment,
                "probability": probability,
                "confidence": confidence,
                "response_time_ms": response_time,
                "timestamp": datetime.utcnow().isoformat(),
            }
        },
    )
    print(f"logged prediction for text: {text} with sentiment: {sentiment}(confiance: {confidence}, temps: {response_time}ms)")
    # print(f"Prediction logged: {sentiment} (confidence: {confidence:.2%}, time: {response_time:.0f}ms)")

def log_error_pred(text: str, predicted_sentiment: str, correct_sentiment: str, confidence: float=0.0, environment="local", probability: float=0.0):
    """Enregistrer une erreur de prédiction dans les logs Azure."""
    logger.error(
        "MISPREDICTION REPORTED BY USER",
        extra={ # info pour l'analyse
            "custom_dimensions": {
                "event_type": "prediction_error",
                "environment": environment,
                "tweet_text": text, 
                "predicted_sentiment": predicted_sentiment,
                "correct_sentiment": correct_sentiment,
                'model_confidence': confidence,   
                "timestamp": datetime.utcnow().isoformat(),
            }
        },
    )
    print(f"Erreur prediction rapportée par l'utilisateur")
    print(f"   Tweet: '{text}'")
    print(f"   Prédit: {predicted_sentiment} ")
    print(f"   Devrait être: {correct_sentiment} ")
    print(f"   Confiance du modèle: {confidence:.2%}")

def log_error(error_type: str, error_message: str, text: str = "", environment="local"):
    """Logger les erreurs système (pas les mal-prédits)"""
    logger.error(
        f"System Error: {error_type}",
        extra={
            "custom_dimensions": {
                "environment": environment,
                "error_type": error_type,
                "error_message": error_message,
                "text_length": len(text) if text else 0,
                "timestamp": datetime.utcnow().isoformat(),
            }
        },
    )
    print(f"Erreur système: {error_type}")

def log_api_health(status: str, environment="local"):
    """Logger l'état de santé de l'API"""
    logger.info(
        "API health check",
        extra={
            "custom_dimensions": {
                "environment": environment,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
            }
        },
    )
    color = "✅" if status == "healthy" else "❌"
    print(f"{color} Health check: {status}")