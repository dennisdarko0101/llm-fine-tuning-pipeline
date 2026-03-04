"""Integration tests for the inference API (no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from src.inference.api import _state, configure_app, create_app
from src.inference.predictor import GenerationConfig, Predictor, PredictionResult


@pytest.fixture()
def mock_predictor() -> MagicMock:
    """Create a mock predictor for API tests."""
    predictor = MagicMock(spec=Predictor)
    predictor.device = "cpu"

    # Mock predict
    predictor.predict.return_value = PredictionResult(
        prompt="test prompt",
        generated_text="This is a generated response.",
        num_tokens=6,
        generation_time=0.5,
    )

    # Mock predict_batch
    predictor.predict_batch.return_value = [
        PredictionResult(
            prompt="prompt 1",
            generated_text="Response 1",
            num_tokens=3,
            generation_time=0.3,
        ),
        PredictionResult(
            prompt="prompt 2",
            generated_text="Response 2",
            num_tokens=3,
            generation_time=0.3,
        ),
    ]

    # Mock predict_stream
    predictor.predict_stream.return_value = iter(["This ", "is ", "streamed."])

    return predictor


@pytest.fixture()
def client(mock_predictor: MagicMock) -> TestClient:
    """Create a test client with configured app."""
    app = create_app()
    # Configure state directly
    _state.predictor = mock_predictor
    _state.model_name = "test-model"
    _state.max_seq_length = 2048
    _state.model_params = {"total_params": 1000000}
    return TestClient(app)


@pytest.fixture()
def client_no_model() -> TestClient:
    """Create a test client without a loaded model."""
    app = create_app()
    _state.predictor = None
    _state.model_name = ""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_with_model(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_without_model(self, client_no_model: TestClient) -> None:
        response = client_no_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_model"
        assert data["model_loaded"] is False


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info(self, client: TestClient) -> None:
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test-model"
        assert data["device"] == "cpu"
        assert data["max_seq_length"] == 2048

    def test_model_info_no_model(self, client_no_model: TestClient) -> None:
        response = client_no_model.get("/model/info")
        assert response.status_code == 503


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict(self, client: TestClient, mock_predictor: MagicMock) -> None:
        response = client.post("/predict", json={"prompt": "Hello world"})
        assert response.status_code == 200
        data = response.json()
        assert data["generated_text"] == "This is a generated response."
        assert data["num_tokens"] == 6
        assert data["generation_time"] > 0
        assert data["tokens_per_second"] > 0

    def test_predict_no_model(self, client_no_model: TestClient) -> None:
        response = client_no_model.post("/predict", json={"prompt": "Hello"})
        assert response.status_code == 503

    def test_predict_empty_prompt(self, client: TestClient) -> None:
        response = client.post("/predict", json={"prompt": ""})
        assert response.status_code == 422  # Validation error

    def test_predict_custom_params(self, client: TestClient, mock_predictor: MagicMock) -> None:
        response = client.post("/predict", json={
            "prompt": "Hello",
            "max_new_tokens": 128,
            "temperature": 0.5,
            "top_p": 0.8,
            "do_sample": False,
        })
        assert response.status_code == 200
        # Verify predictor was called with GenerationConfig
        mock_predictor.predict.assert_called_once()
        call_kwargs = mock_predictor.predict.call_args
        config = call_kwargs[1]["config"]
        assert config.max_new_tokens == 128
        assert config.temperature == 0.5


class TestPredictBatchEndpoint:
    """Tests for /predict/batch endpoint."""

    def test_batch_predict(self, client: TestClient) -> None:
        response = client.post("/predict/batch", json={
            "prompts": ["Hello", "World"],
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["total_time"] > 0

    def test_batch_predict_no_model(self, client_no_model: TestClient) -> None:
        response = client_no_model.post("/predict/batch", json={
            "prompts": ["Hello"],
        })
        assert response.status_code == 503


class TestPredictStreamEndpoint:
    """Tests for /predict/stream endpoint."""

    def test_stream_predict(self, client: TestClient) -> None:
        response = client.post("/predict/stream", json={"prompt": "Hello"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        # Read streamed content
        content = response.text
        assert "This " in content or "streamed" in content

    def test_stream_no_model(self, client_no_model: TestClient) -> None:
        response = client_no_model.post("/predict/stream", json={"prompt": "Hello"})
        assert response.status_code == 503


class TestConfigureApp:
    """Tests for app configuration."""

    def test_configure_app(self) -> None:
        predictor = MagicMock()
        predictor.device = "cpu"
        configure_app(
            predictor,
            model_name="my-model",
            max_seq_length=4096,
            model_params={"total": 7000000},
        )
        assert _state.predictor is predictor
        assert _state.model_name == "my-model"
        assert _state.max_seq_length == 4096
        assert _state.model_params["total"] == 7000000
