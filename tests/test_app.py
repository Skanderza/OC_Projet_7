import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch  
from app.app import app  


class TestAPI(unittest.TestCase):
    """Tests endpoints"""

    @classmethod
    def setUpClass(cls):
        
        cls.client = TestClient(app) # on crée un client de test FastAPI

    def test_index_route(self):
        """
        Vérifie que la route '/' répond bien et renvoie les infos attendues.
        """
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200) 

        data = response.json()
        self.assertIn("message", data) 
        self.assertIn("endpoints", data)
        self.assertEqual(data["message"], "Analyse de sentiment des tweets")
        self.assertIn("/health", data["endpoints"])
        self.assertIn("/predict", data["endpoints"])
        self.assertIn("/feedback", data["endpoints"])

    def test_health_route(self):
        """
        Vérifie que /health renvoie status=ok quand le modèle est chargé.
        Si le modèle n'est pas chargé, l'API renverra 503.
        """
        response = self.client.get("/health")

        # Deux cas possibles -> on teste les deux de manière souple
        self.assertIn(response.status_code, (200, 503))

        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["model"], "logreg_tfidf")
            self.assertIn("environment", data)

    def test_predict_route_with_valid_tweet(self):
        """
        Test sur /predict :
        on envoie un tweet et on vérifie que la réponse est cohérente.
        """
        payload = {"text": "I love this airline, it was a great flight!"}
        response = self.client.post("/predict", json=payload)

        # L'API doit répondre 200 si ok
        self.assertEqual(response.status_code, 200)

        data = response.json()

        # On vérifie la structure de la réponse
        self.assertIn("text", data)
        self.assertIn("sentiment", data)
        self.assertIn("probability", data)
        self.assertIn("confidence", data)

        self.assertEqual(data["text"], payload["text"])
        self.assertIn(data["sentiment"], ("positive", "negative"))

        self.assertGreaterEqual(data["probability"], 0.0)
        self.assertLessEqual(data["probability"], 1.0)
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)

    def test_predict_route_missing_text_field(self):
        """
        Si on n'envoie pas le champ 'text', FastAPI doit répondre 422 (erreur de validation).
        """
        payload = {}  # pas de champ "text"
        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 422)  # erreur de validation Pydantic
        
    @patch("app.app.log_error_pred")
    def test_feedback_success(self, mock_log_error_pred):
        """
        Quand on envoie un feedback correct, l'API doit :
        - répondre 200
        - renvoyer {"status": "success", ...}
        - appeler log_error_pred avec les bons arguments
        """
        payload = {
            "text": "This prediction was wrong",
            "predicted_sentiment": "positive",
            "correct_sentiment": "negative",
            "confidence": 0.85
        }

        response = self.client.post("/feedback", json=payload)

        # 1) Statut HTTP
        self.assertEqual(response.status_code, 200)

        # 2) Contenu de la réponse
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("message", data)

        # 3) Vérifier que log_error_pred a bien été appelé
        mock_log_error_pred.assert_called_once_with(
            text=payload["text"],
            predicted_sentiment=payload["predicted_sentiment"],
            correct_sentiment=payload["correct_sentiment"],
            confidence=payload["confidence"],
            environment="local",
        )

    @patch("app.app.log_error_pred")
    def test_feedback_failure_when_logging_raises(self, mock_log_error_pred):
        """
        Si log_error_pred lève une exception, l'API doit renvoyer 500.
        """
        # On simule une erreur côté logging
        mock_log_error_pred.side_effect = Exception("Boom")

        payload = {
            "text": "This prediction was wrong",
            "predicted_sentiment": "positive",
            "correct_sentiment": "negative",
            "confidence": 0.85
        }

        response = self.client.post("/feedback", json=payload)

        self.assertEqual(response.status_code, 500)

        data = response.json()
        # Le message exact vient de HTTPException dans app.py
        self.assertIn("Erreur lors de l'enregistrement", data["detail"])

# test ok

if __name__ == "__main__":
    unittest.main()
