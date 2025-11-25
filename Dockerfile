# Image de base Python 3.11 optimisée
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie les fichiers de dépendances
COPY requirements.txt .

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger les données NLTK nécessaires
RUN python -m nltk.downloader wordnet \
    && python -m nltk.downloader omw-1.4

# Copie le code de l'application
COPY app/ ./app/

# Copy du modele
COPY models_final/ ./models_final/

# Commande pour démarrer l'API
# CMD uvicorn app.app:app --host 0.0.0.0 --port $PORT
# CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["sh", "-c", "uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-8000}"]

