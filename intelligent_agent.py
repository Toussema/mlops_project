# app_agent.py (VERSION INTELLIGENTE AVEC IA)
"""
Agent intelligent basé sur LangChain pour choisir entre Transformer et TF-IDF
"""

import time
import httpx
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from prometheus_client import Counter, Histogram, CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest

# LangChain
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate

# URLs modèles
TRANSFORMER_URL = "http://transformer_service:8000/predict"
TFIDF_URL = "http://tfidf_service:8001/predict"

# Prometheus
registry = CollectorRegistry()
AGENT_REQUESTS = Counter("agent_requests_total", "Requests to agent", ["model"], registry=registry)
AGENT_LATENCY = Histogram("agent_latency_seconds", "Agent latency", registry=registry)

app = FastAPI(title="AI Model Router", version="2.0")

# IA : LLM + Embeddings
llm = OllamaLLM(model="llama3")  # tu peux mettre mistral, mixtral, phi3-mini, …
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


class TicketInput(BaseModel):
    text: str
    prefer_fast: bool = False


class AgentResponse(BaseModel):
    category: str
    model_used: str
    confidence: float
    latency: float
    reasoning: str


# ---- IA DECISIONNELLE : CHOIX DU MODÈLE ------------------------------------

decision_prompt = PromptTemplate.from_template("""
Tu es un agent intelligent de routage de modèles de machine learning.

Tu dois choisir ENTRE :
- "tfidf"  (rapide mais moins précis)
- "transformer" (lent mais très précis)

Analyse le texte suivant :

Texte : "{text}"
Nombre de mots : {word_count}
Embedding norm : {embedding_norm}
Utilisateur préfère rapidité ? {prefer_fast}

Règles générales :
- Textes très courts / simples → TF-IDF
- Textes techniques, ambigus ou longs → Transformer
- Si prefer_fast=true → privilégie TF-IDF
- Évalue aussi la complexité sémantique grâce aux embeddings.

Répond STRICTEMENT au format JSON :
{{
 "model": "...",
 "reason": "explication en une phrase"
}}
""")


async def choose_model_ai(text: str, prefer_fast: bool) -> dict:
    """Sélection du modèle via IA (LangChain + enchanced reasoning)"""
    embedding = embeddings.embed_query(text)

    prompt = decision_prompt.format(
        text=text,
        word_count=len(text.split()),
        embedding_norm=round(sum(abs(x) for x in embedding), 2),
        prefer_fast=prefer_fast,
    )

    decision = llm.invoke(prompt)

    import json
    try:
        result = json.loads(decision)
    except:
        # fallback
        result = {"model": "tfidf", "reason": "Erreur parsing JSON, fallback TF-IDF"}

    return result


# ---- ROUTAGE ---------------------------------------------------------------

async def call_model(url: str, text: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json={"text": text})
        r.raise_for_status()
        return r.json()


@app.post("/predict", response_model=AgentResponse)
async def predict(ticket: TicketInput):

    start = time.time()

    # IA décide du meilleur modèle
    decision = await choose_model_ai(ticket.text, ticket.prefer_fast)
    chosen = decision["model"]
    reasoning = decision["reason"]

    try:
        url = TRANSFORMER_URL if chosen == "transformer" else TFIDF_URL
        result = await call_model(url, ticket.text)

        prediction = result.get("prediction", result.get("category"))
        confidence = result.get("confidence", 0.80 if chosen == "transformer" else 0.65)

        latency = time.time() - start
        AGENT_REQUESTS.labels(model=chosen).inc()
        AGENT_LATENCY.observe(latency)

        return AgentResponse(
            category=prediction,
            model_used=chosen.upper(),
            confidence=confidence,
            latency=latency,
            reasoning=reasoning,
        )

    except Exception as e:
        # fallback intelligent
        fallback = "tfidf" if chosen == "transformer" else "transformer"
        fallback_url = TFIDF_URL if chosen == "transformer" else TRANSFORMER_URL

        try:
            result = await call_model(fallback_url, ticket.text)
            latency = time.time() - start

            AGENT_REQUESTS.labels(model="fallback").inc()
            AGENT_LATENCY.observe(latency)

            return AgentResponse(
                category=result.get("prediction", result.get("category")),
                model_used=f"FALLBACK_{fallback.upper()}",
                confidence=0.50,
                latency=latency,
                reasoning=f"Primary model failed ({str(e)}), fallback used",
            )
        except:
            latency = time.time() - start
            return AgentResponse(
                category="Error",
                model_used="none",
                confidence=0,
                latency=latency,
                reasoning="Both models failed",
            )


# METRICS
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
