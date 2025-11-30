# Étape 1 : Build des dépendances
FROM python:3.12-slim AS builder

WORKDIR /app

# Mettre à jour pip
RUN pip install --no-cache-dir --upgrade pip

# Installer PyTorch CPU depuis l'index PyTorch
RUN pip install --no-cache-dir --user \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Installer les dépendances de l'API (directement, sans requirements.txt)
RUN pip install --no-cache-dir --user \
    fastapi==0.119.0 \
    uvicorn[standard]==0.38.0 \
    prometheus-client==0.23.1 \
    transformers==4.57.1

# Étape 2 : Image finale légère
FROM python:3.12-slim

WORKDIR /app

# Copier les dépendances installées
COPY --from=builder /root/.local /root/.local

# Ajouter au PATH
ENV PATH=/root/.local/bin:$PATH

# Variables d'environnement
ENV TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1 \
    TRANSFORMERS_NO_JAX=1 \
    PYTHONUNBUFFERED=1

# Copier le code de l'application
COPY app_transformer.py /app/app_transformer.py

# Copier UNIQUEMENT le modèle nécessaire
COPY models/transformer /app/models/transformer

# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER root

# Healthcheck simplifié (sans requests qui n'est pas installé)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/metrics')"

EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "app_transformer:app", "--host", "0.0.0.0", "--port", "8000"]