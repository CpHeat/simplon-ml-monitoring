"""
Scheduler pour exécuter le monitoring de drift toutes les heures.
Expose également les métriques de drift vers Prometheus.
"""
import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from prometheus_client import start_http_server, Gauge, Counter, Info
from drift_monitor import DriftMonitor
import logging

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ===== MÉTRIQUES PROMETHEUS =====

# Drift detection
dataset_drift_detected = Gauge(
    'ml_dataset_drift_detected',
    'Whether dataset drift was detected (1=yes, 0=no)',
    ['model']
)

drift_share = Gauge(
    'ml_drift_share',
    'Share of drifted features (0.0 to 1.0)',
    ['model']
)

drifted_columns_count = Gauge(
    'ml_drifted_columns_count',
    'Number of columns with detected drift',
    ['model']
)

# Performance metrics (pour Titanic seulement)
model_accuracy = Gauge(
    'ml_model_accuracy',
    'Model accuracy based on true labels',
    ['model']
)

samples_with_labels = Gauge(
    'ml_samples_with_labels',
    'Number of samples with true labels available',
    ['model']
)

# Predictions count
predictions_count = Gauge(
    'ml_predictions_count_last_period',
    'Number of predictions in the last monitoring period',
    ['model']
)

# Monitoring runs
monitoring_runs_total = Counter(
    'ml_monitoring_runs_total',
    'Total number of monitoring runs',
    ['model', 'status']
)

# Last run info
last_monitoring_run = Gauge(
    'ml_last_monitoring_run_timestamp',
    'Timestamp of last monitoring run',
    ['model']
)

monitoring_info = Info(
    'ml_monitoring',
    'Information about drift monitoring service'
)

# Set initial info
monitoring_info.info({
    'version': '1.0',
    'frequency': 'hourly',
    'models': 'titanic_ml,titanic_dl,unemployment'
})


class DriftScheduler:
    """Scheduler pour le monitoring de drift"""

    def __init__(self, interval_hours: int = 1):
        """
        Args:
            interval_hours: Intervalle en heures entre les runs
        """
        self.interval_hours = interval_hours
        self.monitor = DriftMonitor()
        self.scheduler = BackgroundScheduler()

    def update_prometheus_metrics(self, results: dict):
        """Mettre à jour les métriques Prometheus avec les résultats"""
        for model_name, metrics in results.items():
            if metrics.get('status') == 'no_data':
                logger.warning(f"No data for {model_name}, skipping metrics update")
                predictions_count.labels(model=model_name).set(0)
                continue

            if metrics.get('status') == 'error':
                logger.error(f"Error for {model_name}: {metrics.get('error')}")
                monitoring_runs_total.labels(model=model_name, status='error').inc()
                continue

            # Drift metrics
            dataset_drift_detected.labels(model=model_name).set(
                1 if metrics.get('dataset_drift_detected', False) else 0
            )
            drift_share.labels(model=model_name).set(
                metrics.get('drift_share', 0.0)
            )
            drifted_columns_count.labels(model=model_name).set(
                metrics.get('number_of_drifted_columns', 0)
            )

            # Predictions count
            predictions_count.labels(model=model_name).set(
                metrics.get('count', 0)
            )

            # Performance metrics (si disponible)
            if 'accuracy' in metrics:
                model_accuracy.labels(model=model_name).set(
                    metrics['accuracy']
                )
                samples_with_labels.labels(model=model_name).set(
                    metrics.get('samples_with_labels', 0)
                )

            # Last run timestamp
            last_monitoring_run.labels(model=model_name).set(
                time.time()
            )

            # Success counter
            monitoring_runs_total.labels(model=model_name, status='success').inc()

            logger.info(f"Updated Prometheus metrics for {model_name}")

    def run_monitoring(self):
        """Exécuter le monitoring et mettre à jour Prometheus"""
        logger.info("="*60)
        logger.info("Starting scheduled drift monitoring run")
        logger.info("="*60)

        try:
            # Exécuter le monitoring
            # Analyse une fenêtre légèrement plus large que l'intervalle (x2) pour avoir assez de données
            # Pour les tests : analyser les dernières 24h pour capturer toutes les données
            analysis_window = max(self.interval_hours * 2, 24)  # Minimum 24h pour les tests
            results = self.monitor.run_all_reports(hours=analysis_window)

            # Mettre à jour Prometheus
            self.update_prometheus_metrics(results)

            logger.info("Monitoring run completed successfully")

        except Exception as e:
            logger.error(f"Error during monitoring run: {e}", exc_info=True)

    def start(self, run_immediately: bool = True):
        """
        Démarrer le scheduler

        Args:
            run_immediately: Si True, exécute un run immédiatement au démarrage
        """
        logger.info(f"Starting drift monitoring scheduler (interval: {self.interval_hours}h)")

        # Run immédiat si demandé
        if run_immediately:
            logger.info("Running initial monitoring check...")
            self.run_monitoring()

        # Programmer les exécutions futures
        # Utiliser IntervalTrigger pour supporter des intervalles < 1h
        from apscheduler.triggers.interval import IntervalTrigger

        self.scheduler.add_job(
            self.run_monitoring,
            trigger=IntervalTrigger(hours=self.interval_hours),
            id='drift_monitoring',
            name='Drift Monitoring Job',
            replace_existing=True
        )

        self.scheduler.start()
        logger.info(f"Scheduler started. Next run: {self.scheduler.get_jobs()[0].next_run_time}")

    def stop(self):
        """Arrêter le scheduler"""
        logger.info("Stopping scheduler...")
        self.scheduler.shutdown()
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    # Port pour les métriques Prometheus
    PROMETHEUS_PORT = 8002
    INTERVAL_HOURS = 0.25  # 15 minutes

    logger.info("="*60)
    logger.info("DRIFT MONITORING SERVICE")
    logger.info("="*60)
    logger.info(f"Prometheus metrics port: {PROMETHEUS_PORT}")
    logger.info(f"Monitoring interval: {INTERVAL_HOURS}h ({int(INTERVAL_HOURS * 60)} minutes)")
    logger.info("="*60)

    # Démarrer le serveur Prometheus
    start_http_server(PROMETHEUS_PORT)
    logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
    logger.info(f"Metrics available at: http://localhost:{PROMETHEUS_PORT}/metrics")

    # Créer et démarrer le scheduler
    scheduler = DriftScheduler(interval_hours=INTERVAL_HOURS)
    scheduler.start(run_immediately=True)

    logger.info("Service is running. Press Ctrl+C to stop.")

    try:
        # Garder le service actif
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("\nShutdown requested...")
        scheduler.stop()
        logger.info("Service stopped")
