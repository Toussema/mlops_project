# app_tfidf.py

"""
API FastAPI pour le modèle TF-IDF + SVM
Endpoint de prédiction avec métriques Prometheus
"""

import time

from fastapi import FastAPI
from fastapi.responses import Response
from joblib import load
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

# ---------------------------------------------------
# 🔹 Création d'un registre Prometheus dédié
# ---------------------------------------------------
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "tfidf_request_count", "Total number of prediction requests", ["category"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "tfidf_request_latency_seconds", "Prediction request latency", registry=registry
)
PREDICTION_CONFIDENCE = Histogram(
    "tfidf_prediction_confidence", "Prediction confidence scores", registry=registry
)

# ---------------------------------------------------
# 🔹 Initialisation de l'application FastAPI
# ---------------------------------------------------
app = FastAPI(
    title="TF-IDF SVM Ticket Classifier API",
    description="Classification de tickets avec TF-IDF + SVM",
    version="1.0.0",
)

# ---------------------------------------------------
# 🔹 Chargement du modèle TF-IDF + SVM
# ---------------------------------------------------
MODEL_PATH = "/app/models/tfidf/tfidf_svm_best.pkl"

print("🚀 Chargement du modèle TF-IDF + SVM...")
model = load(MODEL_PATH)
print("✅ Modèle chargé avec succès !")

# Vérifier les classes disponibles
if hasattr(model, "steps"):
    for name, step in model.steps:
        if hasattr(step, "classes_"):
            print(f"📊 Classes détectées: {list(step.classes_)}")


# ---------------------------------------------------
# 🔹 Schéma d'entrée et sortie
# ---------------------------------------------------
class TicketInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    category: str
    confidence: float
    latency: float
    model: str = "TF-IDF + SVM"


# ---------------------------------------------------
# 🔹 Endpoint de prédiction
# ---------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
async def predict(ticket: TicketInput):
    start_time = time.time()

    try:
        # Prédiction (retourne directement le string)
        prediction = model.predict([ticket.text])[0]

        # Calculer la confiance via decision_function
        if hasattr(model, "decision_function"):
            decision_scores = model.decision_function([ticket.text])[0]
            # Confidence = score maximum normalisé
            max_score = max(decision_scores)
            confidence = float(1.0 / (1.0 + abs(max_score)))  # Normalisation simple
        else:
            confidence = 1.0  # Pas de score de confiance disponible

        latency = time.time() - start_time

        # Métriques Prometheus
        REQUEST_COUNT.labels(category=prediction).inc()
        REQUEST_LATENCY.observe(latency)
        PREDICTION_CONFIDENCE.observe(confidence)

        return PredictionOutput(category=prediction, confidence=confidence, latency=latency)

    except Exception as e:
        REQUEST_LATENCY.observe(time.time() - start_time)
        REQUEST_COUNT.labels(category="error").inc()
        return PredictionOutput(category="Error", confidence=0.0, latency=time.time() - start_time)


# ---------------------------------------------------
# 🔹 Endpoint Prometheus
# ---------------------------------------------------
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------
# 🔹 Health check
# ---------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "model": "TF-IDF + SVM", "version": "1.0.0"}


# ---------------------------------------------------
# 🔹 Informations sur le modèle
# ---------------------------------------------------
@app.get("/model/info")
async def model_info():
    """Retourne les informations sur le modèle"""
    info = {
        "model_type": "TF-IDF + LinearSVC Pipeline",
        "model_path": MODEL_PATH,
    }

    # Extraire les classes disponibles
    if hasattr(model, "steps"):
        for name, step in model.steps:
            if hasattr(step, "classes_"):
                info["classes"] = list(step.classes_)
                info["n_classes"] = len(step.classes_)

    return info
