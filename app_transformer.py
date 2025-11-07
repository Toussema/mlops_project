import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

import time

import torch
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------
# 🔹 Création d'un registre Prometheus dédié
# ---------------------------------------------------
registry = CollectorRegistry()

REQUEST_COUNT = Counter("request_count", "Total number of prediction requests", registry=registry)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Prediction request latency", registry=registry
)

# ---------------------------------------------------
# 🔹 Initialisation de l’application FastAPI
# ---------------------------------------------------
app = FastAPI(title="Transformer Ticket Classifier API")

# ---------------------------------------------------
# 🔹 Chargement du modèle local Hugging Face (Windows)
# ---------------------------------------------------
# ⚠️ Mettez ici le chemin complet vers le dossier transformer sur votre machine
MODEL_PATH = "/app/models/transformer"  # chemin Docker correct

# Mappage par défaut (au cas où le modèle n’en contient pas)
LABEL_MAP = {
    0: "Access",
    1: "Administrative rights",
    2: "HR Support",
    3: "Hardware",
    4: "Internal Project",
    5: "Miscellaneous",
    6: "Purchase",
    7: "Storage",
}

print("🚀 Chargement du modèle local...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
print("✅ Modèle chargé avec succès !")

# Si le modèle a un mapping interne, on l’utilise
if hasattr(model.config, "id2label") and model.config.id2label:
    LABEL_MAP = model.config.id2label
    print("🧭 Mapping du modèle utilisé :", LABEL_MAP)


# ---------------------------------------------------
# 🔹 Schéma d’entrée
# ---------------------------------------------------
class TicketInput(BaseModel):
    text: str


# ---------------------------------------------------
# 🔹 Endpoint de prédiction
# ---------------------------------------------------
@app.post("/predict")
async def predict(ticket: TicketInput):
    REQUEST_COUNT.inc()
    start_time = time.time()

    inputs = tokenizer(
        ticket.text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    label = LABEL_MAP.get(pred, "Unknown")

    REQUEST_LATENCY.observe(time.time() - start_time)
    return {"prediction": label}


# ---------------------------------------------------
# 🔹 Endpoint Prometheus
# ---------------------------------------------------
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
