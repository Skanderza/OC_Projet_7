
import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import tensorflow as tf
from transformers import TFDistilBertModel 


# Chemin vers ton modèle stocké sur MLflow
MODEL_PATH = "/Users/skanderzahi/Desktop/P7/projet_7/mlruns/379503310426968982/models/m-bae25b59f2d94b2b82a0e174e5d4d303/artifacts/data/model"
# Chemin de sortie pour le modèle TFLite
OUTPUT_PATH = "app/model.tflite"

print("chargement du modèle")
custom_objects = {
    "TFDistilBertModel": TFDistilBertModel  # Clé = nom string, Valeur = classe
}
# Chargement du modèle TensorFlow
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
print("Modèle chargé")
print(f"Type du modèle : {type(model)}")


print("creer le convertisseur")
# Convertisseur TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

print("Optimisation du modèle")
# Optimisations pour réduire la taille
# converter.optimizations = [tf.lite.Optimize.DEFAULT] => degrade les performances

# Force float32 (pas de quantification)
converter.target_spec.supported_types = [tf.float32]

print("Executer la conversion du modèle")
# Conversion
tflite_model = converter.convert()

# Sauvegarde
with open(OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)

# Vérifie la taille
original_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                   for dirpath, dirnames, filenames in os.walk(MODEL_PATH)
                   for filename in filenames) / (1024 * 1024)
tflite_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

print(f"Conversion réussie!")
print(f"Taille originale: {original_size:.2f} MB")
print(f"Taille TFLite: {tflite_size:.2f} MB")
print(f"Réduction: {((original_size - tflite_size) / original_size * 100):.1f}%")
print(f"Modèle sauvegardé dans: {OUTPUT_PATH}")




