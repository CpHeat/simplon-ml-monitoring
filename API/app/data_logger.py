"""
Module pour logger les prédictions dans la base de données.
Permet la capture des données pour l'analyse de drift avec Evidently.
"""
import asyncio
from datetime import datetime
from sqlalchemy.orm import Session
from app.database import TitanicPrediction, UnemploymentPrediction, get_db_session


class DataLogger:
    """Classe pour logger les prédictions de manière asynchrone"""

    @staticmethod
    async def log_titanic_prediction(
        model_type: str,
        sex: int,
        pclass: int,
        age: float,
        prediction: int,
        probability_survive: float,
        probability_death: float,
        embarked: int = None,
        true_label: int = None
    ):
        """
        Logger une prédiction Titanic (ML ou DL)

        Args:
            model_type: 'ml' ou 'dl'
            sex: 0=homme, 1=femme
            pclass: 1, 2, 3
            age: Age du passager
            prediction: 0=décès, 1=survie
            probability_survive: Probabilité de survie
            probability_death: Probabilité de décès
            embarked: Port d'embarquement (0=C, 1=Q, 2=S) - optionnel
            true_label: Vrai label si disponible (0=décès, 1=survie) - optionnel
        """
        def _log():
            db = get_db_session()
            try:
                prediction_record = TitanicPrediction(
                    model_type=model_type,
                    sex=sex,
                    pclass=pclass,
                    age=age,
                    embarked=embarked,
                    prediction=prediction,
                    probability_survive=probability_survive,
                    probability_death=probability_death,
                    true_label=true_label
                )
                db.add(prediction_record)
                db.commit()
                print(f"[LOGGER] Titanic {model_type.upper()} prediction logged (ID: {prediction_record.id})")
            except Exception as e:
                print(f"[LOGGER ERROR] Failed to log Titanic prediction: {e}")
                db.rollback()
            finally:
                db.close()

        # Exécuter en thread séparé pour ne pas bloquer l'API
        await asyncio.to_thread(_log)

    @staticmethod
    async def log_unemployment_prediction(
        country: str,
        year: int,
        agriculture: float,
        industry: float,
        services: float,
        gdp_log: float,
        predicted_unemployment_rate: float
    ):
        """
        Logger une prédiction de taux de chômage

        Args:
            country: Nom du pays
            year: Année
            agriculture: Part agriculture (%)
            industry: Part industrie (%)
            services: Part services (%)
            gdp_log: Log du PIB
            predicted_unemployment_rate: Taux de chômage prédit
        """
        def _log():
            db = get_db_session()
            try:
                prediction_record = UnemploymentPrediction(
                    country=country,
                    year=year,
                    agriculture=agriculture,
                    industry=industry,
                    services=services,
                    gdp_log=gdp_log,
                    predicted_unemployment_rate=predicted_unemployment_rate
                )
                db.add(prediction_record)
                db.commit()
                print(f"[LOGGER] Unemployment prediction logged (ID: {prediction_record.id}, Country: {country})")
            except Exception as e:
                print(f"[LOGGER ERROR] Failed to log Unemployment prediction: {e}")
                db.rollback()
            finally:
                db.close()

        # Exécuter en thread séparé pour ne pas bloquer l'API
        await asyncio.to_thread(_log)


# Instance singleton
data_logger = DataLogger()
