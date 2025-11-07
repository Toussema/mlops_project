"""
Tests unitaires pour l'API Transformer
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def mock_model():
    """Mock du modèle transformer pour les tests"""
    with (
        patch("app_transformer.AutoModelForSequenceClassification") as mock_model_class,
        patch("app_transformer.AutoTokenizer") as mock_tokenizer_class,
    ):

        # Mock du tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[101, 2054, 2003, 102]],
            "attention_mask": [[1, 1, 1, 1]],
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock du modèle
        mock_model_instance = Mock()
        mock_output = Mock()
        mock_output.logits = [[0.1, 0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]]
        mock_model_instance.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model_instance

        yield mock_tokenizer, mock_model_instance


@pytest.fixture
def client():
    """Client de test FastAPI"""
    # Importer l'app après le mock
    from app_transformer import app

    return TestClient(app)


class TestPredictEndpoint:
    """Tests pour l'endpoint /predict"""

    def test_predict_success(self, client, mock_model):
        """Test de prédiction réussie"""
        response = client.post("/predict", json={"text": "I need access to the database"})

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], str)

    def test_predict_empty_text(self, client):
        """Test avec texte vide"""
        response = client.post("/predict", json={"text": ""})

        assert response.status_code == 200
        assert "prediction" in response.json()

    def test_predict_missing_text(self, client):
        """Test avec champ text manquant"""
        response = client.post("/predict", json={})

        assert response.status_code == 422  # Validation error

    def test_predict_invalid_json(self, client):
        """Test avec JSON invalide"""
        response = client.post(
            "/predict", data="invalid json", headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_predict_long_text(self, client, mock_model):
        """Test avec texte très long"""
        long_text = "test " * 500
        response = client.post("/predict", json={"text": long_text})

        assert response.status_code == 200
        assert "prediction" in response.json()


class TestMetricsEndpoint:
    """Tests pour l'endpoint /metrics"""

    def test_metrics_endpoint(self, client):
        """Test de l'endpoint metrics"""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_counters(self, client):
        """Vérifier que les métriques contiennent les compteurs"""
        # Faire une prédiction d'abord
        client.post("/predict", json={"text": "test"})

        # Récupérer les métriques
        response = client.get("/metrics")
        content = response.text

        assert "request_count" in content
        assert "request_latency_seconds" in content

    def test_metrics_increment(self, client):
        """Vérifier que les métriques s'incrémentent"""
        # Première requête
        response1 = client.get("/metrics")
        content1 = response1.text

        # Faire des prédictions
        for _ in range(3):
            client.post("/predict", json={"text": "test"})

        # Deuxième requête
        response2 = client.get("/metrics")
        content2 = response2.text

        # Les métriques devraient avoir changé
        assert content1 != content2


class TestHealthCheck:
    """Tests pour le healthcheck"""

    def test_root_endpoint(self, client):
        """Test de l'endpoint racine"""
        response = client.get("/")

        # Si l'endpoint existe
        if response.status_code == 200:
            assert response.json() is not None
        else:
            # Si pas d'endpoint root, c'est OK aussi
            assert response.status_code == 404


@pytest.mark.integration
class TestModelIntegration:
    """Tests d'intégration avec le vrai modèle (optionnel)"""

    @pytest.mark.skipif(
        not os.path.exists("models/transformer"), reason="Modèle transformer non disponible"
    )
    def test_real_model_prediction(self):
        """Test avec le vrai modèle (si disponible)"""
        from app_transformer import app

        client = TestClient(app)

        test_cases = [
            "I need VPN access",
            "My laptop is broken",
            "Question about salary",
        ]

        for text in test_cases:
            response = client.post("/predict", json={"text": text})
            assert response.status_code == 200
            assert "prediction" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
