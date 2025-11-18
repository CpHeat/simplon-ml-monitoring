FROM python:3.10-slim

WORKDIR /app

# Copier le fichier requirements.txt
COPY ../requirements.txt .

# Installer les dépendances avec permissions appropriées
RUN pip install --no-cache-dir --user -r requirements.txt

# Ajouter les binaires utilisateur au PATH
ENV PATH="/root/.local/bin:${PATH}"

# Copier le code de l'application
COPY ./app /app/app

# Exposer le port 8000
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
