"""
Main FastAPI Application Entrypoint.
Hospital Appointment No-Show Prediction Backend.
Provides high-performance REST APIs with Swagger documentation.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, metrics, prediction
from api.routes.prediction import get_predictor
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("fastapi_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event to pre-warm and validate ML models on startup."""
    logger.info("Initializing Hospital No-Show Prediction API...")
    try:
        get_predictor()
        logger.info("ML predictor pre-warmed and ready for inference.")
    except Exception as e:
        logger.warning(f"Predictor could not pre-warm on startup: {e}")
    yield
    logger.info("Shutting down API service.")


def create_app() -> FastAPI:
    """Application factory for FastAPI service."""
    config = load_config()

    app = FastAPI(
        title=config["api"].get("title", "Hospital Appointment No-Show Prediction API"),
        version=config["api"].get("version", "1.0.0"),
        description=(
            "### Clinical Operations Decision Support System\n\n"
            "Predicts the probability of patient appointment no-shows using calibrated "
            "machine learning models (XGBoost, Random Forest, etc.) with SHAP explainability.\n\n"
            "- **Prediction Scenario:** Booking-time (Leak-free)\n"
            "- **Risk Tiers:** Low (<30%), Medium (30-60%), High (>=60%)\n"
            "- **Swagger UI:** Accessible at `/docs`"
        ),
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(health.router)
    app.include_router(prediction.router)
    app.include_router(metrics.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
