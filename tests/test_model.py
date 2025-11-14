import unittest

from app.Model import model, tokenizer, predict_sentiment, MAX_LENGTH
from app.Tweets import PredictedResult


@unittest.skipIf(
    model is None or tokenizer is None,
    "Modèle ou tokenizer non chargé"
)
class TestModel(unittest.TestCase):
    """Tests autour du modèle de sentiment."""

    def test_model_and_tokenizer_loaded(self):
        """Vérifie que le modèle et le tokenizer sont bien chargés."""
        # assertIsNotNone : Vérifie que la variable n'est pas None
        self.assertIsNotNone(model, "Le modèle n'a pas été chargé")
        self.assertIsNotNone(tokenizer, "Le tokenizer n'a pas été chargé")
        print("Modèle et tokenizer chargés avec succès.")

    def test_predict_sentiment_basic_output(self):
        """Appelle predict_sentiment sur un texte simple et vérifie la cohérence du résultat."""
        text = "This is a test tweet"
        result = predict_sentiment(text)

        # Type de retour => verifie que result est bien un type PredictedResult
        self.assertIsInstance(result, PredictedResult)
        print("predict_sentiment retourne un PredictedResult")

        # Le texte original est bien conservé
        self.assertEqual(result.text, text)
        print("Le texte original est bien conservé")

        # Sentiment doit être 'positive' ou 'negative'
        self.assertIn(result.sentiment, ("positive", "negative"))
        print("Le sentiment est valide:", result.sentiment)

        # Probabilité et confiance entre 0 et 1
        self.assertGreaterEqual(result.probability, 0.0)
        self.assertLessEqual(result.probability, 1.0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        print("La probabilité et la confiance sont dans l'intervalle [0, 1]")

    def test_predict_sentiment_is_stable(self):
        """Pour un même texte, le modèle doit renvoyer un résultat stable."""
        text = "it's raining!"
        res1 = predict_sentiment(text)
        res2 = predict_sentiment(text)

        self.assertEqual(res1.sentiment, res2.sentiment)
        print("Le sentiment est stable entre deux prédictions")
        self.assertAlmostEqual(res1.probability, res2.probability, places=5)
        print("La probabilité est stable entre deux prédictions")


if __name__ == "__main__":
    unittest.main()
