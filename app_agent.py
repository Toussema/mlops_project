# app_agent.py
"""
Agent de routage intelligent qui choisit entre Transformer et TF-IDF
Basé sur des règles et les performances des modèles
"""

import time

import httpx
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

# Configuration
TRANSFORMER_URL = "http://transformer_service:8000/predict"
TFIDF_URL = "http://tfidf_service:8001/predict"

# Prometheus
registry = CollectorRegistry()

AGENT_REQUESTS = Counter(
    "agent_requests_total", "Total requests to agent", ["model_chosen"], registry=registry
)
AGENT_LATENCY = Histogram("agent_latency_seconds", "Agent routing latency", registry=registry)

app = FastAPI(
    title="MLOps Agent - Model Router",
    description="Agent intelligent qui route vers le meilleur modèle",
    version="1.0.0",
)


class TicketInput(BaseModel):
    text: str
    prefer_fast: bool = False  # Si True, préfère TF-IDF (plus rapide)


class AgentResponse(BaseModel):
    category: str  # ← Changé de "prediction"
    model_used: str
    confidence: float
    latency: float
    reasoning: str


def get_text_complexity(text: str) -> str:
    """Évalue la complexité du texte"""
    word_count = len(text.split())

    if word_count <= 5:
        return "simple"
    elif word_count <= 15:
        return "medium"
    else:
        return "complex"


def choose_model(text: str, prefer_fast: bool = False) -> str:
    """
    Stratégie de choix du modèle:
    - Textes simples → TF-IDF (plus rapide)
    - Textes complexes → Transformer (plus précis)
    - Si prefer_fast=True → TF-IDF
    """

    if prefer_fast:
        return "tfidf"

    complexity = get_text_complexity(text)

    # Règles de routage
    if complexity == "simple":
        return "tfidf"  # TF-IDF suffit pour les textes courts
    elif complexity == "complex":
        return "transformer"  # Transformer pour les textes longs
    else:
        # Pour medium, alterner ou choisir selon le contexte
        # Ici on préfère TF-IDF par défaut (plus rapide)
        return "tfidf"


async def call_model(url: str, text: str) -> dict:
    """Appelle un modèle et retourne la réponse"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json={"text": text})
        response.raise_for_status()
        return response.json()


@app.post("/predict", response_model=AgentResponse)
async def predict(ticket: TicketInput):
    """Route vers le meilleur modèle et retourne la prédiction"""
    start_time = time.time()

    # Choisir le modèle
    model_choice = choose_model(ticket.text, ticket.prefer_fast)
    complexity = get_text_complexity(ticket.text)

    # Construire le raisonnement (E501 fix: ligne découpée)
    reasoning = (
        f"Text complexity: {complexity}, "
        f"Word count: {len(ticket.text.split())}, "
        f"Prefer fast: {ticket.prefer_fast}"
    )

    # Appeler le modèle choisi
    try:
        if model_choice == "transformer":
            url = TRANSFORMER_URL
            result = await call_model(url, ticket.text)
            prediction = result.get("prediction", "Unknown")
            confidence = 0.85  # Placeholder
            model_used = "Transformer"

        else:  # tfidf
            url = TFIDF_URL
            result = await call_model(url, ticket.text)
            prediction = result.get("category", "Unknown")
            confidence = result.get("confidence", 0.0)
            model_used = "TF-IDF + SVM"

        latency = time.time() - start_time

        # Métriques
        AGENT_REQUESTS.labels(model_chosen=model_choice).inc()
        AGENT_LATENCY.observe(latency)

        return AgentResponse(
            category=prediction,  # ← Changé
            model_used=model_used,
            confidence=confidence,
            latency=latency,
            reasoning=reasoning,
        )

    except Exception as e:  # E722 FIX
        # Fallback vers l'autre modèle si échec
        try:
            fallback_url = TFIDF_URL if model_choice == "transformer" else TRANSFORMER_URL
            result = await call_model(fallback_url, ticket.text)

            latency = time.time() - start_time
            AGENT_REQUESTS.labels(model_chosen="fallback").inc()
            AGENT_LATENCY.observe(latency)

            return AgentResponse(
                prediction=result.get("prediction", result.get("category", "Error")),
                model_used=f"Fallback ({'TF-IDF' if model_choice == 'transformer' else 'Transformer'})",
                confidence=0.5,
                latency=latency,
                reasoning=f"Primary model failed: {str(e)}, used fallback",
            )

        except Exception as fallback_error:  # E722 FIX
            latency = time.time() - start_time
            AGENT_REQUESTS.labels(model_chosen="error").inc()
            AGENT_LATENCY.observe(latency)

            return AgentResponse(
                prediction="Error",
                model_used="None",
                confidence=0.0,
                latency=latency,
                reasoning=f"All models failed: {str(fallback_error)}",
            )

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    """Check health of agent and both models"""
    health_status = {"agent": "healthy", "models": {}}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check Transformer
            try:
                resp = await client.get("http://transformer_service:8000/metrics")
                health_status["models"]["transformer"] = "up" if resp.status_code == 200 else "down"
            except:
                health_status["models"]["transformer"] = "down"

            # Check TF-IDF
            try:
                resp = await client.get("http://tfidf_service:8001/health")
                health_status["models"]["tfidf"] = "up" if resp.status_code == 200 else "down"
            except:
                health_status["models"]["tfidf"] = "down"
    except:
        health_status["models"] = {"transformer": "unknown", "tfidf": "unknown"}

    return health_status


@app.get("/stats")
async def stats():
    """Retourne des statistiques sur l'utilisation des modèles"""
    return {
        "routing_strategy": "complexity-based",
        "models_available": ["Transformer", "TF-IDF + SVM"],
        "rules": {
            "simple_text": "TF-IDF (fast)",
            "complex_text": "Transformer (accurate)",
            "prefer_fast": "TF-IDF",
        },
    }
