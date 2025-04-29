from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_keywords():
    data = {
        "text": "Le machine learning permet de prédire des résultats.",
        "max_keywords": 3
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "keywords" in result
    assert len(result["keywords"]) <= 3
