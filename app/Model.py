from tensorflow import keras
from transformers import TFDistilBertModel, DistilBertTokenizer
from Tweets import PredictedResult  
import os

os.environ['TF_USE_LEGACY_KERAS'] = '0'

# Chemin  vers le modèle dans MLflow
MODEL_PATH = "/Users/skanderzahi/Desktop/P7/projet/mlruns/379503310426968982/models/m-bae25b59f2d94b2b82a0e174e5d4d303/artifacts/data/model"
MAX_LENGTH = 128

# Initialisation des variables
model = None
tokenizer = None

try:
    # Charger le modèle 
    print("Chargement du modèle")
    model = keras.models.load_model(MODEL_PATH,
                                    custom_objects={"TFDistilBertModel": TFDistilBertModel})
    # Charger le tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    print("Modèle + tokenizer chargés")
except Exception as e:
    print(f"Erreur lors du chargement: {e}")
    model = None
    tokenizer = None

def predict_sentiment(text: str) -> PredictedResult:
    if model is None or tokenizer is None:
        raise ValueError("Modèle ou tokenizer non chargé")
    
    # Tokenization et padding
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    
    # Prédiction
    prediction = model.predict(
        [encoding['input_ids'], encoding['attention_mask']],
        verbose=0
    )
    

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
    