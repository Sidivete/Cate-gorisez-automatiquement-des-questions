# tests/test_app.py

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Bienvenue sur l'API de prédiction de mots-clés" in response.json()["message"]
    assert "/predict" in response.json()["endpoints"]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_ready"] is True
    assert response.json()["bert_ready"] is True

def test_predict_keywords():
    data = {
        "text": "Le machine learning permet de prédire des résultats.",
        "max_keywords": 3
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    
    # Vérifie que la réponse contient des mots-clés
    assert "keywords" in result
    assert isinstance(result["keywords"], list)
    assert len(result["keywords"]) <= data["max_keywords"]
    
    # Vérifie que les probabilités sont bien présentes et arrondies à 3 décimales
    assert "probabilities" in result
    assert len(result["probabilities"]) == len(result["keywords"])
    assert all(isinstance(prob, float) and round(prob, 3) == prob for prob in result["probabilities"])
