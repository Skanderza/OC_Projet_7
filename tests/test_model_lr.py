import unittest

from app.Model_LR import tfidf, logreg, predict_sentiment_lr
from app.Tweets import PredictedResult

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@unittest.skipIf(
    tfidf is None or logreg is None,
    "TF-IDF ou LogisticRegression non chargé(s)"
)
class TestModelLR(unittest.TestCase):
    """Tests autour du modèle TF-IDF + Logistic Regression."""

    def test_tfidf_and_logreg_loaded(self):
        """
        Vérifie que le TF-IDF et la LogisticRegression sont bien chargés
        et du bon type.
        """
        self.assertIsNotNone(tfidf, "Le vectorizer TF-IDF n'a pas été chargé")
        self.assertIsNotNone(logreg, "Le modèle LogisticRegression n'a pas été chargé")

        self.assertIsInstance(tfidf, TfidfVectorizer)
        self.assertIsInstance(logreg, LogisticRegression)

    def test_tfidf_logreg_compatible(self):
        """
        Vérifie que le nombre de features du TF-IDF
        correspond bien à ce que la LogisticRegression attend.
        """
        X = tfidf.transform(["test sentence for checking dimensions"])
        n_features = X.shape[1]

        self.assertEqual(
            n_features,
            logreg.n_features_in_,
            f"Incompatibilité TF-IDF / LogReg : TF-IDF={n_features}, "
            f"LogReg attend={logreg.n_features_in_}",
        )

    def test_predict_sentiment_lr_basic_output(self):
        """
        Appelle predict_sentiment_lr sur un texte simple
        et vérifie la cohérence du résultat.
        """
        text = "I love this movie"
        result = predict_sentiment_lr(text)

        # Type de retour
        self.assertIsInstance(result, PredictedResult)

        # Le texte original est bien conservé
        self.assertEqual(result.text, text)

        # Sentiment doit être 'positive' ou 'negative'
        self.assertIn(result.sentiment, ("positive", "negative"))

        # Probabilité et confiance entre 0 et 1
        self.assertGreaterEqual(result.probability, 0.0)
        self.assertLessEqual(result.probability, 1.0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_predict_sentiment_lr_stability(self):
        """
        Pour un même texte, le modèle doit renvoyer un résultat stable.
        (même label et proba quasi identique)
        """
        text = "What a bad movie"

        res1 = predict_sentiment_lr(text)
        res2 = predict_sentiment_lr(text)

        self.assertEqual(res1.sentiment, res2.sentiment)
        self.assertAlmostEqual(res1.probability, res2.probability, places=6)


if __name__ == "__main__":
    unittest.main()
