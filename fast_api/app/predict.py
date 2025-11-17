# app/predict.py
import asyncio
import numpy as np
from app.model_loader import model_titanic_ml, scaler_X_titanic_ml

class Predict:
    async def predict_survive_ml(self, data: dict, genre: str):
        genre = genre.lower()
        if genre not in ["homme", "femme"]:
            raise ValueError(f"Genre inconnu : {genre} (seules 'homme' et 'femme' sont autorisées)")
        
        print(f"Données reçues: {data}")
        model = "LogisticRegression"
        
        possible_features = {
            "sex": 1 if genre == "femme" else 0,  # ✅ Encodage correct : femme=1, homme=0
        }
        
        def _predict():
            nonlocal model
            feature_to_use = ["sex"]
            X = np.array([possible_features[feature] for feature in feature_to_use]).reshape(1, -1)
            print(f"Features Titanic ({feature_to_use}) - Shape: {X.shape}, Valeurs: {X}")
            X_scaled = scaler_X_titanic_ml.transform(X)
            prediction = model_titanic_ml.predict(X_scaled)[0]
            proba = model_titanic_ml.predict_proba(X_scaled)[0]
            
            return {
                "prediction": int(prediction),  # 0=décès, 1=survie
                "probability_survive": float(proba[1]),
                "probability_death": float(proba[0]),
                "model": model,
                "input_data": data
            }
        
        try:
            return await asyncio.to_thread(_predict)
        except KeyError as e:
            raise ValueError(f"Clé manquante dans les données : {e}")
        except Exception as e:
            print(f"🔥 Erreur dans predict_survive_ml: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ValueError("Erreur interne lors de la prédiction. Vérifiez les champs saisis.")