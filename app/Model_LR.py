import os
import joblib
from .Tweets import PredictedResult  
import re 
import nltk
from nltk.tokenize import TweetTokenizer
# from nltk.stem import WordNetLemmatizer



# Preprocess de l'entrainement

tweet_tokenizer = TweetTokenizer(
    preserve_case=False,   # uniformisation en minuscules
    reduce_len=True,       # réduit les répétitions de caractères 
    strip_handles=False,  
)
# lemmatizer = WordNetLemmatizer()

# Tokenisation spécifique aux tweets
def tweetTokenize(text: str):
    tokens = tweet_tokenizer.tokenize(text)
    return tokens

def preprocess_2(text:str)-> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text) # Supprimer les URLs
    text = re.sub(r"\d+", "", text) # Supprimer les chiffres
    text = re.sub(r"\@\w+|\#", "", text) # Supprimer les mentions et hashtags (@user, #hashtag)
    text = re.sub(r"\b[a-z]\b", "", text) # Supprimer mots d'1 lettre
    tokens = tweetTokenize(text) # Tokenisation
    # tokens = [lemmatizer.lemmatize(word) for word in tokens] # Lemmatisation 
    return " ".join(tokens)

# Chargement du modele

# Dossier courant = app/
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models_lr")

VECT_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer_133k.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_tfidf_133k.joblib")

tfidf = None
logreg = None

try:
    print("Chargement TF-IDF + LogisticRegression")

    tfidf = joblib.load(VECT_PATH)
    logreg = joblib.load(MODEL_PATH)

    print("  TF-IDF chargé :", type(tfidf))
    print("  LogisticRegression chargée :", type(logreg))

except Exception as e:
    print(f"Erreur lors du chargement du modèle LR : {e}")
    tfidf = None
    logreg = None
    
# Prediction

def predict_sentiment_lr(text: str) -> PredictedResult:
    
    if tfidf is None or logreg is None:
        raise ValueError("Modèle LR ou TF-IDF non chargé")
    
    text_clean = preprocess_2(text)

    # Transformer le texte en vecteur TF-IDF
    X = tfidf.transform([text_clean])

    # Probabilité de la classe "positive" 
    proba_pos = float(logreg.predict_proba(X)[:, 1][0])

    sentiment = "positive" if proba_pos > 0.5 else "negative"
    confidence = proba_pos if proba_pos > 0.5 else (1 - proba_pos)

    return PredictedResult(
        text=text,
        sentiment=sentiment,
        probability=round(proba_pos, 4),
        confidence=round(confidence, 4),
    )
