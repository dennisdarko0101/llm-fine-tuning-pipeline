"""FastAPI inference server with predict, streaming, and batch endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""

    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_new_tokens: int = Field(256, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(50, ge=0, description="Top-k sampling")
    do_sample: bool = Field(True, description="Whether to use sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")


class PredictResponse(BaseModel):
    """Response body for /predict endpoint."""

    generated_text: str
    num_tokens: int
    generation_time: float
    tokens_per_second: float


class BatchPredictRequest(BaseModel):
    """Request body for /predict/batch endpoint."""

    prompts: list[str] = Field(..., min_length=1, max_length=32, description="List of prompts")
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    do_sample: bool = Field(True)


class BatchPredictResponse(BaseModel):
    """Response body for /predict/batch endpoint."""

    results: list[PredictResponse]
    total_time: float


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response body for /model/info endpoint."""

    model_name: str
    device: str
    max_seq_length: int
    parameters: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------
# App state
# ---------------------------------------------------------------


class _AppState:
    """Mutable application state for the FastAPI app."""

    def __init__(self) -> None:
        self.predictor: Any = None
        self.model_name: str = ""
        self.max_seq_length: int = 2048
        self.model_params: dict[str, Any] = {}


_state = _AppState()


# ---------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LLM Fine-Tuning Inference API",
        description="API for generating text with fine-tuned language models",
        version="1.0.0",
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if _state.predictor is not None else "no_model",
            model_loaded=_state.predictor is not None,
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        """Get loaded model information."""
        if _state.predictor is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        return ModelInfoResponse(
            model_name=_state.model_name,
            device=_state.predictor.device,
            max_seq_length=_state.max_seq_length,
            parameters=_state.model_params,
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        """Generate text for a single prompt."""
        if _state.predictor is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        from src.inference.predictor import GenerationConfig

        config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
            repetition_penalty=request.repetition_penalty,
        )
        result = _state.predictor.predict(request.prompt, config=config)
        return PredictResponse(
            generated_text=result.generated_text,
            num_tokens=result.num_tokens,
            generation_time=round(result.generation_time, 4),
            tokens_per_second=round(result.tokens_per_second, 2),
        )

    @app.post("/predict/stream")
    async def predict_stream(request: PredictRequest) -> StreamingResponse:
        """Stream generated text token-by-token."""
        if _state.predictor is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        from src.inference.predictor import GenerationConfig

        config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
            repetition_penalty=request.repetition_penalty,
        )

        def stream_generator():
            yield from _state.predictor.predict_stream(request.prompt, config=config)

        return StreamingResponse(stream_generator(), media_type="text/plain")

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
        """Generate text for multiple prompts."""
        if _state.predictor is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        import time

        from src.inference.predictor import GenerationConfig

        config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
        )

        start = time.time()
        results = _state.predictor.predict_batch(request.prompts, config=config)
        total_time = time.time() - start

        return BatchPredictResponse(
            results=[
                PredictResponse(
                    generated_text=r.generated_text,
                    num_tokens=r.num_tokens,
                    generation_time=round(r.generation_time, 4),
                    tokens_per_second=round(r.tokens_per_second, 2),
                )
                for r in results
            ],
            total_time=round(total_time, 4),
        )

    return app


def configure_app(
    predictor: Any,
    model_name: str = "",
    max_seq_length: int = 2048,
    model_params: dict[str, Any] | None = None,
) -> None:
    """Configure the app with a predictor instance.

    Args:
        predictor: Predictor instance for generation.
        model_name: Model name for reporting.
        max_seq_length: Maximum sequence length.
        model_params: Model parameter stats.
    """
    _state.predictor = predictor
    _state.model_name = model_name
    _state.max_seq_length = max_seq_length
    _state.model_params = model_params or {}
    log.info("api_configured", model=model_name)


# Create default app instance
app = create_app()
