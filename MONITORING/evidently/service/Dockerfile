FROM python:3.11-slim

WORKDIR /app

# Copier seulement les requirements du service (pas ceux de l'API !)
COPY ./MONITORING/evidently/service/requirements.txt .

# Installer uniquement les dépendances nécessaires au drift monitoring
RUN pip install --no-cache-dir -r requirements.txt

# Créer le dossier app pour les modules partagés
RUN mkdir -p /app/app

# Copier le module database.py depuis l'API (nécessaire pour accéder à la DB)
COPY ./API/app/database.py /app/app/database.py

# Créer un __init__.py vide pour que app soit un module Python
RUN echo "" > /app/app/__init__.py

# Copier le code du service
COPY ./MONITORING/evidently/service/drift_monitor.py ./
COPY ./MONITORING/evidently/service/scheduler.py ./

# Créer les dossiers pour les volumes
RUN mkdir -p /app/reports /app/logs

WORKDIR /app

# Exposer le port pour les métriques Prometheus
EXPOSE 8002

# Lancer le scheduler
CMD ["python", "scheduler.py"]
