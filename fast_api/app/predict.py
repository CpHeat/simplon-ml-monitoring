# app/predict.py
import asyncio
import numpy as np
from app.model_loader import model_titanic_ml, scaler_X_titanic_ml,scaler_X_titanic_dl,model_titanic_dl,le_sex,le_embarked 

class Predict:
    async def predict_survive_ml(self, data: dict, genre: str,pclass: int,age: int):
        genre = genre.lower()
        if genre not in ["homme", "femme"]:
            raise ValueError(f"Genre inconnu : {genre} (seules 'homme' et 'femme' sont autorisées)")
        if pclass not in [1,2,3]:
            raise ValueError(f"pclasse inconnu : {genre} (seules '1','2','3' sont autorisées)")

        print(f"Données reçues: {data}")
        model = "LogisticRegression"
        
        possible_features = {
            "sex": 1 if genre == "femme" else 0,  # Encodage correct : femme=1, homme=0
            "pclass": pclass,
            "age":age
        }
        
        def _predict():
            nonlocal model
            feature_to_use = ["sex","pclass","age"]
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
        

# ===== Deep Learning (nouveau) =====
    async def predict_survive_dl(self, data: dict, genre: str, pclass: int, age: int, embarked: str):
        genre = genre.lower()
        embarked = embarked.upper()  # Normalise en majuscule (C, Q, S)
        
        if genre not in ["homme", "femme"]:
            raise ValueError(f"Genre inconnu : {genre}")
        if embarked not in ["C", "Q", "S"]:
            raise ValueError(f"Port d'embarquement inconnu : {embarked} (attendu: C, Q, S)")
        
        print(f"Genre: {genre}, Pclass: {pclass}, Age: {age}, Embarked: {embarked}")
        model_name = "DeepLearning"
        
        def _predict():
            nonlocal model_name
            
            sex_mapping = {
            "homme": "male",
            "femme": "female"
            }
            sex_en = sex_mapping[genre]  # Conversion
            # Encodage des features catégorielles avec LabelEncoder
            sex_encoded = le_sex.transform([sex_en])[0]  # "homme" ou "femme" → 0 ou 1
            embarked_encoded = le_embarked.transform([embarked])[0]  # C/Q/S → 0/1/2
            
            # Construction du vecteur de features (ajuste l'ordre selon l'entraînement !)
            # Exemple : [pclass, sex, age, embarked]
            X = np.array([[pclass, sex_encoded, age, embarked_encoded]])
            
            print(f"Features DL [pclass, sex, age, embarked]: {X}")
            
            # Scaling
            X_scaled = scaler_X_titanic_dl.transform(X)
            
            # Prédiction (DL retourne une proba entre 0 et 1)
            proba = model_titanic_dl.predict(X_scaled, verbose=0)[0][0]  # Proba de survie
            prediction = 1 if proba >= 0.5 else 0  # Seuil à 0.5
            
            return {
                "prediction": int(prediction),
                "probability_survive": float(proba),
                "probability_death": float(1 - proba),
                "model": model_name,
                "input_data": {
                    "genre": genre,
                    "pclass": pclass,
                    "age": age,
                    "embarked": embarked
                }
            }
        
        try:
            return await asyncio.to_thread(_predict)
        except Exception as e:
            print(f"🔥 Erreur dans predict_survive_dl: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ValueError("Erreur interne lors de la prédiction DL.")