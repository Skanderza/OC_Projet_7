# Image de base Python 3.11 optimisée
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie les fichiers de dépendances
COPY requirements.txt .

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie le code de l'application
COPY app/ ./app/

# Expose le port (Heroku l'assignera dynamiquement)
EXPOSE $PORT

# Commande pour démarrer l'API
CMD uvicorn app.app:app --host 0.0.0.0 --port $PORT
