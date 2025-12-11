"""
Service de monitoring de drift avec Evidently AI.
Analyse les prédictions stockées et génère des rapports de drift.
"""
import os
import sys
import json
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

        titanic_ref_path = REFERENCE_DIR / "titanic_reference.csv"
        unemployment_ref_path = REFERENCE_DIR / "unemployment_reference.csv"

        self.reference_titanic = pd.read_csv(titanic_ref_path)        
        self.reference_unemployment = pd.read_csv(unemployment_ref_path)        

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
        Générer un rapport de drift pour Titanic avec métriques de performance.

        Returns:
            Dict avec métriques de drift et chemin du rapport
        """

        current_data, count = self.get_recent_titanic_predictions(hours, model_type)

        if count == 0:
            return {"status": "no_data", "count": 0}

        # Colonnes communes pour ML et DL
        feature_columns = ['sex', 'pclass', 'age']
        if model_type == 'dl':
            feature_columns.append('embarked')

        # Filtrer les colonnes du référence pour correspondre
        reference_data = self.reference_titanic[feature_columns + ['survived']].copy()
        current_data_filtered = current_data[feature_columns + ['survived']].dropna()

        # Configuration Evidently avec DataDefinition
        categorical_cols = [c for c in feature_columns if c != 'age']
        categorical_cols.append('survived')
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

        # Créer le rapport de drift complet
        report = Report(metrics=[
            DataDriftPreset()
        ])

        # Exécuter avec la nouvelle API (current, reference)
        my_eval = report.run(current_dataset, reference_dataset)

        # Sauvegarder le rapport HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"titanic_{model_type}_drift_{timestamp}.html"
        my_eval.save_html(str(report_path))
        print(f"  Report saved: {report_path.name}")

        # Extraire les métriques
        report_dict = json.loads(my_eval.json())
        metrics = self._extract_metrics_from_report(report_dict)
        metrics['model'] = f'titanic_{model_type}'
        metrics['count'] = count
        metrics['report_path'] = str(report_path)

        # Si on a des true_labels, calculer les métriques de performance
        if 'true_label' in current_data.columns and current_data['true_label'].notna().sum() > 0:
            print(f"  Calculating performance metrics...")
            perf_metrics = self._calculate_detailed_performance_metrics(current_data)
            metrics.update(perf_metrics)

            # Créer un rapport HTML custom pour les performances
            perf_report_path = REPORTS_DIR / f"titanic_{model_type}_performance_{timestamp}.html"
            self._save_performance_html(perf_metrics, perf_report_path)
            metrics['performance_report_path'] = str(perf_report_path)
            print(f"  Performance report saved: {perf_report_path.name}")

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
            'drifted_features': [],
            'feature_drift_details': {},
            'target_drift_detected': False
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

                        # Le value contient les détails du drift
                        value = metric.get('value', {})

                        if isinstance(value, dict):
                            drift_detected = value.get('drift_detected', False)
                            drift_score = value.get('drift_score', 1.0)

                            # Stocker les détails par feature
                            metrics['feature_drift_details'][column_name] = {
                                'drift_detected': drift_detected,
                                'drift_score': drift_score
                            }

                            # Ajouter à la liste des features driftées
                            if drift_detected:
                                metrics['drifted_features'].append(column_name)

                        # Fallback si value est un simple nombre (p-value)
                        elif isinstance(value, (int, float)):
                            drift_detected = value < 0.05
                            metrics['feature_drift_details'][column_name] = {
                                'drift_detected': drift_detected,
                                'drift_score': value
                            }
                            if drift_detected:
                                metrics['drifted_features'].append(column_name)

                # Détecter le target drift (si la colonne "survived", "unemployment_rate" ou "target" drifte)
                elif 'TargetDrift' in metric_name or ('ValueDrift' in metric_name and ('survived' in metric_name.lower() or 'unemployment_rate' in metric_name.lower())):
                    value = metric.get('value', {})
                    if isinstance(value, dict):
                        metrics['target_drift_detected'] = value.get('drift_detected', False)
                    elif isinstance(value, (int, float)):
                        metrics['target_drift_detected'] = value < 0.05

        except Exception as e:
            print(f"  [WARNING] Error extracting metrics: {e}")

        # Recalculer le drift_share en excluant le target
        # feature_drift_details contient SEULEMENT les features (pas le target)
        if metrics['feature_drift_details']:
            total_features = len(metrics['feature_drift_details'])
            drifted_features_count = sum(1 for info in metrics['feature_drift_details'].values() if info.get('drift_detected', False))

            if total_features > 0:
                # Recalculer le drift share basé SEULEMENT sur les features
                metrics['drift_share'] = drifted_features_count / total_features
                metrics['number_of_drifted_columns'] = drifted_features_count
                metrics['dataset_drift_detected'] = metrics['drift_share'] > 0.5

                print(f"  [DRIFT RECALC] {drifted_features_count}/{total_features} features drifted ({metrics['drift_share']:.2%})")

        return metrics

    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculer les métriques de performance basiques (compatibilité)"""
        return self._calculate_detailed_performance_metrics(data)

    def _calculate_detailed_performance_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculer des métriques de performance détaillées : Accuracy, Precision, Recall, F1"""
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

        # Calcul de TP, TN, FP, FN
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['precision'] = float(precision)

        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['recall'] = float(recall)

        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics['f1_score'] = float(f1)

        print(f"  [PERFORMANCE] Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

        return metrics

    def _save_performance_html(self, metrics: Dict, output_path: Path):
        """Créer un rapport HTML custom pour les métriques de performance"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 800px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 8px;
            color: white;
            text-align: center;
        }}
        .metric-card.green {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .metric-card.blue {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        .metric-card.orange {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}
        .metric-card.red {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .metric-value {{
            font-size: 48px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 18px;
            opacity: 0.9;
        }}
        .confusion-matrix {{
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        .timestamp {{
            color: #666;
            font-size: 14px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Model Performance Report</h1>
        <p><strong>Model:</strong> {metrics.get('model', 'Unknown')}</p>
        <p><strong>Samples:</strong> {metrics.get('samples_with_labels', 0)}</p>

        <div class="metric-grid">
            <div class="metric-card green">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{metrics.get('accuracy', 0):.3f}</div>
            </div>
            <div class="metric-card blue">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{metrics.get('f1_score', 0):.3f}</div>
            </div>
            <div class="metric-card orange">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{metrics.get('precision', 0):.3f}</div>
            </div>
            <div class="metric-card red">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{metrics.get('recall', 0):.3f}</div>
            </div>
        </div>

        <div class="confusion-matrix">
            <h2>Confusion Matrix</h2>
            <table>
                <thead>
                    <tr>
                        <th></th>
                        <th>Predicted: No Survive (0)</th>
                        <th>Predicted: Survive (1)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <th>Actual: No Survive (0)</th>
                        <td style="background-color: #c8e6c9;">TN: {metrics.get('true_negatives', 0)}</td>
                        <td style="background-color: #ffcdd2;">FP: {metrics.get('false_positives', 0)}</td>
                    </tr>
                    <tr>
                        <th>Actual: Survive (1)</th>
                        <td style="background-color: #ffcdd2;">FN: {metrics.get('false_negatives', 0)}</td>
                        <td style="background-color: #c8e6c9;">TP: {metrics.get('true_positives', 0)}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="timestamp">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

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
