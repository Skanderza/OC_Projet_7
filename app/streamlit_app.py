
import streamlit as st
import requests
from PIL import Image


st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Logo
logo = Image.open("/Users/skanderzahi/Desktop/P7/projet_7/app/logo.webp")

col1, col2 = st.columns([1, 4]) # Crée 2 colonnes avec ratio 1:4
with col1:
    st.image(logo, width=100)
    # st.logo(logo)
with col2:
    st.title("_Air paradis_")

# URL de l'API
API_URL = "http://127.0.0.1:8000"

# Titre principal
st.title("Tweet Sentiment Analysis")
st.markdown("---")


# Vérifier si l'API est accessible
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


# Afficher le statut de l'API
if check_api_health():
    st.success("API connected")
else:
    st.error("API not connected")
    st.stop()

# Zone de texte pour le tweet
st.subheader("Enter your tweet")
tweet_text = st.text_area(
    "Tweet text", placeholder="Your tweet", height=100, max_chars=128
)

# Compteur de caractères
st.caption(f"Characters: {len(tweet_text)}/128")

# Bouton de prédiction
st.markdown("---")
if st.button("Analyze sentiment", type="tertiary", use_container_width=True):
    if not tweet_text or len(tweet_text.strip()) == 0:
        st.warning("Please enter a tweet")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict", json={"text": tweet_text}, timeout=10
                )

                if response.status_code == 200:
                    result = response.json()

                    # Sauvegarder résultats dans session_state
                    # Permet de garder l'affichage même après re-run
                    st.session_state["show_results"] = True
                    st.session_state["last_prediction"] = {
                        "text": result["text"],
                        "sentiment": result["sentiment"],
                        "probability": result["probability"],
                        "confidence": result["confidence"],
                    }
                    st.rerun()  # Forcer le re-run pour afficher les résultats

                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                st.error("Timeout: API is taking too long to respond")
            except requests.exceptions.ConnectionError:
                st.error("Connection error: Please ensure the API is running")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Afficher résultats EN DEHORS du bloc button
# Si 'show_results' est True dans session_state
if (
    st.session_state.get("show_results", False)
    and "last_prediction" in st.session_state
):

    result = st.session_state["last_prediction"]
    st.markdown("---")

    # Affichage des résultats
    # Sentiment 
    sentiment = result["sentiment"]
    color = "green" if sentiment == "positive" else "red"
    st.markdown(f"### :{color}[{sentiment.capitalize()} sentiment]")

    # Métriques
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Probability", value=f"{result['probability']:.2%}")

    with col2:
        st.metric(label="Confidence", value=f"{result['confidence']:.2%}")

    # Barre de progression
    st.progress(result["confidence"])

    # Feedback 
    st.markdown("---")
    st.markdown("Was this prediction correct?")

    # Widget feedback avec clé unique 
    feedback = st.feedback("thumbs", key=f"feedback_{result['text'][:20]}")
    # retour
    if feedback == 1:
        st.success("Thank you for your feedback. You selected: :material/thumb_up: prediction")
    elif feedback == 0:
        if not st.session_state.get("feedback_sent", False):
            # st.session_state["feedback_sent"] = True  # Pour éviter les multiples envois
            try:
                feedback_response = requests.post(
                    f"{API_URL}/feedback",
                    json={
                        "text": result["text"],
                        "predicted_sentiment": result["sentiment"],
                        "correct_sentiment": "negative" if result["sentiment"] == "positive" else "positive",
                        "confidence": result["confidence"]
                    }, timeout=10
                )
                if feedback_response.status_code == 200:
                    st.session_state["feedback_sent"] = True  # Pour éviter les multiples envois
                    
                    st.error("Thank you for your feedback. You selected: :material/thumb_down: prediction")
                    st.markdown("We'll use your feedback to improve the model.")
                else:
                    st.error(f"Error sending feedback: {feedback_response.status_code}")
            except requests.exceptions.Timeout:
                st.error("Timeout sending feedback")
            except Exception as e:
                    st.error(f"Error: {str(e)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:   
                    st.info("Feedback already sent.")



# Footer
st.markdown("---")
st.caption("Powered by DistilBERT | FastAPI | Streamlit | Air Paradis © 2025")
