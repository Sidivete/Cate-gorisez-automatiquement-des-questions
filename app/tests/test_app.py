from fastapi.testclient import TestClient
from app.main import app

def test_predict_keywords():
    with TestClient(app) as client:  # 👈 Ce "with" est important
        data = {
            "text": "Le machine learning permet de prédire des résultats.",
            "max_keywords": 3
        }
        response = client.post("/predict", json=data)
        print("DEBUG:", response.status_code, response.json())
         # Afficher le statut et le contenu de la réponse
        print(f"Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")  # Affiche les prédictions dans les logs
        assert response.status_code == 200
