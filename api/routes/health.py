"""
Health check and system status routes for FastAPI.
"""
from datetime import datetime, timezone
from fastapi import APIRouter
from api.schemas import HealthResponse
from src.utils.config import get_resolved_path, load_config

router = APIRouter(tags=["System & Health"])


@router.get("/", summary="Root System Status")
def root_endpoint():
    """Return API name, version, and documentation link."""
    config = load_config()
    return {
        "service": config["api"]["title"],
        "version": config["api"]["version"],
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health", response_model=HealthResponse, summary="System Health Verification")
def health_endpoint():
    """Verify application health and availability of serialized model artifacts."""
    config = load_config()
    models_dir = get_resolved_path(config["paths"]["models_dir"])
    model_file = models_dir / "final_model.joblib"
    prep_file = models_dir / "preprocessor.joblib"

    is_ready = model_file.exists() and prep_file.exists()
    status_str = "healthy" if is_ready else "degraded (model artifacts missing)"

    return HealthResponse(
        status=status_str,
        service=config["api"]["title"],
        version=config["api"]["version"],
        model_loaded=is_ready,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
