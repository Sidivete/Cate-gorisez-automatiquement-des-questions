from fastapi import FastAPI, HTTPException
from joblib import load
import numpy as np
from pydantic import BaseModel
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Configuration globale
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "model_supervise_proba.pkl"

# Modèles Pydantic
class TextInput(BaseModel):
    text: str
    max_keywords: int = 5

class KeywordOutput(BaseModel):
    keywords: List[str]
    probabilities: List[float]

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # 1. Charger le modèle BERT pour la vectorisation
        app.state.bert = SentenceTransformer('all-MiniLM-L6-v2')

        # 2. Charger le classifieur
        app.state.clf = load(MODEL_PATH)

        # 3. Charger les tags
        from app.models.all_tags import all_tags
        app.state.tags = all_tags

        # Vérification des dimensions
        test_embedding = app.state.bert.encode(["test"])
        if test_embedding.shape[1] != 384:
            raise ValueError("Dimension incorrecte des embeddings BERT")

        _ = app.state.clf.predict_proba(test_embedding)
        print("✅ Modèles chargés avec succès")

    except Exception as e:
        raise RuntimeError(f"ERREUR: {str(e)}\n"
                           "Vérifiez que:\n"
                           "1. Le modèle BERT est installé (pip install sentence-transformers)\n"
                           "2. Le classifieur attend bien des embeddings BERT\n"
                           "3. Les dimensions des modèles sont compatibles")

# Initialisation FastAPI
app = FastAPI(lifespan=lifespan)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=KeywordOutput)
async def predict_keywords(data: TextInput):
    try:
        embedding = app.state.bert.encode([data.text])
        if embedding.shape[0] != 1 or embedding.shape[1] != 384:
            raise ValueError("Format d'embedding incorrect")

        probabilities = app.state.clf.predict_proba(embedding)[0]
        top_indices = np.argsort(probabilities)[-data.max_keywords:][::-1]
        top_probs = [float(probabilities[i]) for i in top_indices]

        return {
            "keywords": [app.state.tags[i] for i in top_indices],
            "probabilities": [round(p, 3) for p in top_probs]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_ready": hasattr(app.state, 'clf'),
        "bert_ready": hasattr(app.state, 'bert')
    }

@app.get("/")
async def root():
    return {
        "message": "Bienvenue sur l'API de prédiction de mots-clés",
        "endpoints": {
            "/docs": "Documentation interactive",
            "/predict": "POST: Prédire des mots-clés",
            "/health": "GET: Vérifier l'état de l'API",
            "/": "Cette page d'accueil"
        },
        "model_info": {
            "bert": "all-MiniLM-L6-v2",
            "classifier": str(app.state.clf.__class__) if hasattr(app.state, 'clf') else "Non chargé"
        }
    }
