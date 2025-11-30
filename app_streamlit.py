"""
Interface utilisateur Streamlit pour la classification de tickets
"""

import os
import time

import pandas as pd
import requests
import streamlit as st

# Configuration - Utiliser les variables d'environnement pour Docker
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8002") + "/predict"
TRANSFORMER_URL = os.getenv("TRANSFORMER_URL", "http://localhost:8000") + "/predict"
TFIDF_URL = os.getenv("TFIDF_URL", "http://localhost:8001") + "/predict"

# Configuration de la page
st.set_page_config(page_title="MLOps Ticket Classifier", page_icon="🎫", layout="wide")

# Style CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .model-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    .transformer-badge {
        background-color: #ff7f0e;
        color: white;
    }
    .tfidf-badge {
        background-color: #2ca02c;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown('<h1 class="main-header">🎫 Ticket Classifier MLOps</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")
mode = st.sidebar.radio(
    "Mode de prédiction",
    ["Agent Intelligent", "Transformer uniquement", "TF-IDF uniquement", "Comparaison"],
)

prefer_fast = st.sidebar.checkbox("Préférer la rapidité", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Informations")
st.sidebar.info(
    """
**Agent Intelligent**: Route automatiquement vers le meilleur modèle

**Transformer**: Modèle deep learning (précis, lent)

**TF-IDF + SVM**: Modèle classique (rapide, efficace)
"""
)

# Debug info
with st.sidebar.expander("🔧 Configuration réseau"):
    st.text(f"Agent: {AGENT_URL}")
    st.text(f"Transformer: {TRANSFORMER_URL}")
    st.text(f"TF-IDF: {TFIDF_URL}")

# Zone de saisie
st.markdown("### 📝 Entrez votre ticket")
ticket_text = st.text_area(
    "Description du ticket",
    placeholder="Exemple: My laptop screen is broken and I need a replacement...",
    height=150,
)

# Bouton de prédiction
if st.button("🚀 Classifier", type="primary"):
    if not ticket_text.strip():
        st.warning("⚠️ Veuillez entrer un texte de ticket")
    else:
        # Mode Agent
        if mode == "Agent Intelligent":
            with st.spinner("🤖 L'agent analyse le ticket..."):
                try:
                    start_time = time.time()
                    response = requests.post(
                        AGENT_URL,
                        json={"text": ticket_text, "prefer_fast": prefer_fast},
                        timeout=30,
                    )
                    latency = time.time() - start_time

                    if response.status_code == 200:
                        result = response.json()

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### 🎯 Prédiction")
                            st.markdown(
                                f'<div class="prediction-box">'
                                f'<h2 style="color: #1f77b4;">{result["prediction"]}</h2>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        with col2:
                            st.markdown("### 📊 Détails")
                            st.metric("Modèle utilisé", result["model_used"])
                            st.metric("Confiance", f"{result['confidence']:.2%}")
                            st.metric("Latence", f"{result['latency']:.3f}s")

                        with st.expander("🔍 Raisonnement de l'agent"):
                            st.text(result["reasoning"])
                    else:
                        st.error(f"❌ Erreur HTTP: {response.status_code}")
                        st.code(response.text)

                except requests.exceptions.ConnectionError as e:
                    st.error(f"❌ Erreur de connexion à l'agent")
                    st.error(f"URL: {AGENT_URL}")
                    st.error(f"Détails: {str(e)}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Timeout: L'agent met trop de temps à répondre")
                except Exception as e:
                    st.error(f"❌ Erreur inattendue: {type(e).__name__}")
                    st.error(str(e))

        # Mode Transformer
        elif mode == "Transformer uniquement":
            with st.spinner("🤖 Classification avec Transformer..."):
                try:
                    start_time = time.time()
                    response = requests.post(
                        TRANSFORMER_URL, json={"text": ticket_text}, timeout=30
                    )
                    latency = time.time() - start_time

                    if response.status_code == 200:
                        result = response.json()

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### 🎯 Prédiction")
                            st.markdown(
                                f'<div class="prediction-box">'
                                f'<h2 style="color: #ff7f0e;">{result["category"]}</h2>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        with col2:
                            st.markdown("### 📊 Métriques")
                            st.metric("Confiance", f"{result['confidence']:.2%}")
                            st.metric("Latence", f"{result['latency']:.3f}s")
                            st.markdown(
                                '<span class="model-badge transformer-badge">Transformer</span>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.error(f"❌ Erreur: {response.status_code}")

                except Exception as e:
                    st.error(f"❌ Erreur de connexion: {e}")

        # Mode TF-IDF
        elif mode == "TF-IDF uniquement":
            with st.spinner("📊 Classification avec TF-IDF + SVM..."):
                try:
                    start_time = time.time()
                    response = requests.post(TFIDF_URL, json={"text": ticket_text}, timeout=30)
                    latency = time.time() - start_time

                    if response.status_code == 200:
                        result = response.json()

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### 🎯 Prédiction")
                            st.markdown(
                                f'<div class="prediction-box">'
                                f'<h2 style="color: #2ca02c;">{result["category"]}</h2>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        with col2:
                            st.markdown("### 📊 Métriques")
                            st.metric("Confiance", f"{result['confidence']:.2%}")
                            st.metric("Latence", f"{result['latency']:.3f}s")
                            st.markdown(
                                '<span class="model-badge tfidf-badge">TF-IDF + SVM</span>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.error(f"❌ Erreur: {response.status_code}")

                except Exception as e:
                    st.error(f"❌ Erreur de connexion: {e}")

        # Mode Comparaison
        else:  # Comparaison
            st.markdown("### 📊 Comparaison des modèles")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🤖 Transformer")
                with st.spinner("Classification..."):
                    try:
                        start_time = time.time()
                        response = requests.post(
                            TRANSFORMER_URL, json={"text": ticket_text}, timeout=30
                        )
                        latency_transformer = time.time() - start_time

                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"**Prédiction**: {result['category']}")
                            st.metric("Confiance", f"{result['confidence']:.2%}")
                            st.metric("Latence", f"{result['latency']:.3f}s")
                        else:
                            st.error("❌ Échec")
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")

            with col2:
                st.markdown("#### 📊 TF-IDF + SVM")
                with st.spinner("Classification..."):
                    try:
                        start_time = time.time()
                        response = requests.post(TFIDF_URL, json={"text": ticket_text}, timeout=30)
                        latency_tfidf = time.time() - start_time

                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"**Prédiction**: {result['category']}")
                            st.metric("Latence", f"{result['latency']:.3f}s")
                            st.metric("Confiance", f"{result['confidence']:.2%}")
                        else:
                            st.error("❌ Échec")
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666;">
    <p>🔗 <a href="http://localhost:9090" target="_blank">Prometheus</a> | 
       <a href="http://localhost:3000" target="_blank">Grafana</a> | 
       <a href="http://localhost:5000" target="_blank">MLflow</a></p>
</div>
""",
    unsafe_allow_html=True,
)
