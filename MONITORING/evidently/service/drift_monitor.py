"""
Service de monitoring de drift avec Evidently AI.
Analyse les prédictions stockées et génère des rapports de drift.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

# Import database module (available in /app/app/ in the container)
from app.database import get_db_session, TitanicPrediction, UnemploymentPrediction

# Chemins
# Support pour Docker et développement local
REFERENCE_DIR = Path(os.getenv("REFERENCE_DIR", "/app/reference_data"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/app/reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class DriftMonitor:
    """Moniteur de drift pour les modèles ML"""

    def __init__(self):
        self.reference_titanic = None
        self.reference_unemployment = None
        self.load_reference_data()

    def load_reference_data(self):
        """Charger les datasets de référence"""
        print("[DRIFT_MONITOR] Loading reference datasets...")

        titanic_ref_path = REFERENCE_DIR / "titanic_reference.csv"
        unemployment_ref_path = REFERENCE_DIR / "unemployment_reference.csv"

        if titanic_ref_path.exists():
            self.reference_titanic = pd.read_csv(titanic_ref_path)
            print(f"  Titanic reference: {self.reference_titanic.shape}")
        else:
            print(f"  [WARNING] Titanic reference not found: {titanic_ref_path}")

        if unemployment_ref_path.exists():
            self.reference_unemployment = pd.read_csv(unemployment_ref_path)
            print(f"  Unemployment reference: {self.reference_unemployment.shape}")
        else:
            print(f"  [WARNING] Unemployment reference not found: {unemployment_ref_path}")

    def get_recent_titanic_predictions(
        self,
        hours: int = 24,
        model_type: Optional[str] = None
    ) -> Tuple[pd.DataFrame, int]:
        """
        Récupérer les prédictions Titanic récentes

        Args:
            hours: Nombre d'heures en arrière
            model_type: 'ml' ou 'dl' ou None pour tous

        Returns:
            (DataFrame, count)
        """
        db = get_db_session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            query = db.query(TitanicPrediction).filter(
                TitanicPrediction.timestamp >= cutoff_time
            )

            if model_type:
                query = query.filter(TitanicPrediction.model_type == model_type)

            predictions = query.all()
            count = len(predictions)

            if count == 0:
                return pd.DataFrame(), 0

            # Convertir en DataFrame
            data = [{
                'sex': p.sex,
                'pclass': p.pclass,
                'age': p.age,
                'embarked': p.embarked,
                'prediction': p.prediction,
                'probability_survive': p.probability_survive,
                'survived': p.prediction,  # Alias pour target
                'target': p.prediction,
                'true_label': p.true_label,
                'timestamp': p.timestamp
            } for p in predictions]

            df = pd.DataFrame(data)
            return df, count

        finally:
            db.close()

    def get_recent_unemployment_predictions(self, hours: int = 24) -> Tuple[pd.DataFrame, int]:
        """Récupérer les prédictions Unemployment récentes"""
        db = get_db_session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            predictions = db.query(UnemploymentPrediction).filter(
                UnemploymentPrediction.timestamp >= cutoff_time
            ).all()

            count = len(predictions)
            if count == 0:
                return pd.DataFrame(), 0

            data = [{
                'country': p.country,
                'year': p.year,
                'agriculture': p.agriculture,
                'industry': p.industry,
                'services': p.services,
                'gdp_log': p.gdp_log,
                'unemployment_rate': p.predicted_unemployment_rate,
                'target': p.predicted_unemployment_rate,
                'timestamp': p.timestamp
            } for p in predictions]

            df = pd.DataFrame(data)
            return df, count

        finally:
            db.close()

    def generate_titanic_drift_report(self, hours: int = 24, model_type: str = 'ml') -> Dict:
        """
        Générer un rapport de drift pour Titanic

        Returns:
            Dict avec métriques de drift
        """
        print(f"\n[TITANIC {model_type.upper()}] Generating drift report (last {hours}h)...")

        current_data, count = self.get_recent_titanic_predictions(hours, model_type)

        if count == 0:
            print(f"  [WARNING] No predictions in last {hours}h")
            return {"status": "no_data", "count": 0}

        print(f"  Found {count} predictions")

        # Colonnes communes pour ML et DL
        feature_columns = ['sex', 'pclass', 'age']
        if model_type == 'dl':
            feature_columns.append('embarked')

        # Filtrer les colonnes du référence pour correspondre
        reference_data = self.reference_titanic[feature_columns + ['survived']].copy()
        current_data_filtered = current_data[feature_columns + ['survived']].dropna()

        # Configuration Evidently avec DataDefinition (v0.7+)
        # Pour drift detection, on spécifie juste les types de colonnes
        categorical_cols = [c for c in feature_columns if c != 'age']
        categorical_cols.append('survived')  # Le target est aussi catégoriel pour Titanic
        data_definition = DataDefinition(
            numerical_columns=['age'],
            categorical_columns=categorical_cols
        )

        # Créer les Dataset objects
        reference_dataset = Dataset.from_pandas(
            reference_data,
            data_definition=data_definition
        )
        current_dataset = Dataset.from_pandas(
            current_data_filtered,
            data_definition=data_definition
        )

        # Créer le rapport
        # DataDriftPreset inclut le drift des features et du target
        report = Report(metrics=[
            DataDriftPreset()
        ])

        # Exécuter avec la nouvelle API (current, reference)
        # run() retourne un objet Snapshot avec les résultats
        my_eval = report.run(current_dataset, reference_dataset)

        # Sauvegarder le rapport HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"titanic_{model_type}_drift_{timestamp}.html"
        my_eval.save_html(str(report_path))
        print(f"  Report saved: {report_path.name}")

        # Extraire les métriques
        import json
        report_dict = json.loads(my_eval.json())
        metrics = self._extract_metrics_from_report(report_dict)
        metrics['model'] = f'titanic_{model_type}'
        metrics['count'] = count
        metrics['report_path'] = str(report_path)

        # Si on a des true_labels, calculer les métriques de performance
        if 'true_label' in current_data.columns and current_data['true_label'].notna().sum() > 0:
            metrics.update(self._calculate_performance_metrics(current_data))

        return metrics

    def generate_unemployment_drift_report(self, hours: int = 24) -> Dict:
        """Générer un rapport de drift pour Unemployment"""
        print(f"\n[UNEMPLOYMENT] Generating drift report (last {hours}h)...")

        current_data, count = self.get_recent_unemployment_predictions(hours)

        if count == 0:
            print(f"  [WARNING] No predictions in last {hours}h")
            return {"status": "no_data", "count": 0}

        print(f"  Found {count} predictions")

        # Colonnes
        feature_columns = ['agriculture', 'industry', 'services', 'gdp_log', 'year', 'country']

        reference_data = self.reference_unemployment[feature_columns + ['unemployment_rate']].copy()
        current_data_filtered = current_data[feature_columns + ['unemployment_rate']].dropna()

        # Configuration Evidently avec DataDefinition (v0.7+)
        data_definition = DataDefinition(
            numerical_columns=['agriculture', 'industry', 'services', 'gdp_log', 'year'],
            categorical_columns=['country']
            # Pas de classification pour la régression, le target sera détecté automatiquement
        )

        # Créer les Dataset objects
        reference_dataset = Dataset.from_pandas(
            reference_data,
            data_definition=data_definition
        )
        current_dataset = Dataset.from_pandas(
            current_data_filtered,
            data_definition=data_definition
        )

        # Créer le rapport
        # DataDriftPreset inclut le drift des features et du target
        report = Report(metrics=[
            DataDriftPreset()
        ])

        # Exécuter avec la nouvelle API
        my_eval = report.run(current_dataset, reference_dataset)

        # Sauvegarder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"unemployment_drift_{timestamp}.html"
        my_eval.save_html(str(report_path))
        print(f"  Report saved: {report_path.name}")

        # Métriques
        import json
        report_dict = json.loads(my_eval.json())
        metrics = self._extract_metrics_from_report(report_dict)
        metrics['model'] = 'unemployment'
        metrics['count'] = count
        metrics['report_path'] = str(report_path)

        return metrics

    def _extract_metrics_from_report(self, report_dict: dict) -> Dict:
        """Extraire les métriques clés du rapport Evidently (v0.7+)"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'dataset_drift_detected': False,
            'drift_share': 0.0,
            'number_of_drifted_columns': 0,
            'drifted_features': []
        }

        try:
            # Dans v0.7, la structure est différente
            for metric in report_dict.get('metrics', []):
                metric_name = metric.get('metric_name', '')

                # Chercher le metric DriftedColumnsCount qui contient le drift share
                if 'DriftedColumnsCount' in metric_name:
                    value = metric.get('value', {})
                    if isinstance(value, dict):
                        drift_share = value.get('share', 0.0)
                        drifted_count = value.get('count', 0)

                        metrics['drift_share'] = drift_share
                        metrics['number_of_drifted_columns'] = int(drifted_count)
                        metrics['dataset_drift_detected'] = drift_share > 0.5  # Si plus de 50% des features driftent

                # Collecter les features individuelles qui ont drifté
                elif 'ValueDrift' in metric_name:
                    # Extraire le nom de la colonne du metric_name
                    # Format: "ValueDrift(column=age,method=K-S p_value,threshold=0.05)"
                    if 'column=' in metric_name:
                        col_start = metric_name.index('column=') + 7
                        col_end = metric_name.index(',', col_start)
                        column_name = metric_name[col_start:col_end]

                        # Le value est le p-value ou drift score
                        drift_score = metric.get('value', 0)
                        # Si le drift score est faible (p-value < 0.05), la colonne a drifté
                        if isinstance(drift_score, (int, float)) and drift_score < 0.05:
                            metrics['drifted_features'].append(column_name)

        except Exception as e:
            print(f"  [WARNING] Error extracting metrics: {e}")

        return metrics

    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculer les métriques de performance si true_label disponible"""
        metrics = {}

        # Filtrer les lignes avec true_label
        valid_data = data[data['true_label'].notna()].copy()

        if len(valid_data) == 0:
            return metrics

        y_true = valid_data['true_label'].astype(int)
        y_pred = valid_data['prediction'].astype(int)

        # Accuracy
        accuracy = (y_true == y_pred).mean()
        metrics['accuracy'] = float(accuracy)
        metrics['samples_with_labels'] = len(valid_data)

        print(f"  [PERFORMANCE] Accuracy: {accuracy:.3f} ({len(valid_data)} samples)")

        return metrics

    def run_all_reports(self, hours: int = 24) -> Dict[str, Dict]:
        """Exécuter tous les rapports de drift"""
        print(f"\n{'='*60}")
        print(f"DRIFT MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        results = {}

        # Titanic ML
        try:
            results['titanic_ml'] = self.generate_titanic_drift_report(hours, 'ml')
        except Exception as e:
            print(f"[ERROR] Titanic ML: {e}")
            results['titanic_ml'] = {"status": "error", "error": str(e)}

        # Titanic DL
        try:
            results['titanic_dl'] = self.generate_titanic_drift_report(hours, 'dl')
        except Exception as e:
            print(f"[ERROR] Titanic DL: {e}")
            results['titanic_dl'] = {"status": "error", "error": str(e)}

        # Unemployment
        try:
            results['unemployment'] = self.generate_unemployment_drift_report(hours)
        except Exception as e:
            print(f"[ERROR] Unemployment: {e}")
            results['unemployment'] = {"status": "error", "error": str(e)}

        print(f"\n{'='*60}")
        print("SUMMARY:")
        for model, result in results.items():
            status = result.get('status', 'ok')
            count = result.get('count', 0)
            drift = result.get('dataset_drift_detected', False)
            print(f"  {model:20} | {count:4} samples | Drift: {drift}")
        print(f"{'='*60}\n")

        return results


if __name__ == "__main__":
    # Test du monitoring
    monitor = DriftMonitor()
    results = monitor.run_all_reports(hours=168)  # Last 7 days for testing

    print("\nDrift monitoring completed!")
    print(f"Reports saved in: {REPORTS_DIR}")
