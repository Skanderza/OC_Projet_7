import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
from .Tweets import PredictedResult  


# Chemin  vers le modèle dans MLflow
# MODEL_PATH = "/Users/skanderzahi/Desktop/P7/projet_7/mlruns/379503310426968982/models/m-bae25b59f2d94b2b82a0e174e5d4d303/artifacts/data/model"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.tflite") # le modèle TFLite au lieu du modèle Keras complet
MAX_LENGTH = 128

# Initialisation des variables
interpreter = None  # interpreter TFLite au lieu de model Keras
tokenizer = None
input_details = None  # Détails des inputs du modèle TFLite
output_details = None 

try:
    # Charger l'interpréteur TFLite
    print("Chargement du modèle TFLite")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Récupérer les détails d'entrée/sortie
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Charger le tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    print("Modèle + tokenizer chargés")
except Exception as e:
    print(f"Erreur lors du chargement: {e}")
    model = None
    tokenizer = None
    
  
model = interpreter # Pour compatibilité

def predict_sentiment(text: str) -> PredictedResult:
    if interpreter is None or tokenizer is None:
        raise ValueError("Modèle ou tokenizer non chargé")
    
    # Tokenization et padding
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    
    # Préparer les inputs pour TFLite
    input_ids = encoding['input_ids'].astype(np.int32)
    attention_mask = encoding['attention_mask'].astype(np.int32)

    # CHANGEMENT 10 : Utiliser l'interpréteur TFLite pour la prédiction
    interpreter.set_tensor(input_details[0]['index'], input_ids)
    interpreter.set_tensor(input_details[1]['index'], attention_mask)
    
    # Exécuter l'inférence
    interpreter.invoke()
    
    # Prédiction
    prediction = interpreter.get_tensor(output_details[0]['index'])


    # Résultat
    probability = float(prediction[0][0])
    sentiment = "positive" if probability > 0.5 else "negative"
    confidence = probability if probability > 0.5 else (1 - probability)

    return PredictedResult(
        text=text,
        sentiment=sentiment,
        probability=round(probability, 4),
        confidence=round(confidence, 4)
)
    