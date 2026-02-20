from cProfile import label

from fastapi import FastAPI, UploadFile, File, HTTPException
import logging
from src.model import predict_image
from src.schemas import PredictionResponse
from src.config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Containerized ML Prediction API",
    version="2.0.0"
)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Model loaded successfully.")
    yield
    # Shutdown (nothing needed here)

app = FastAPI(
    title="Containerized ML Prediction API",
    version="2.0.0",
    lifespan=lifespan
)


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files (e.g., JPEG, PNG) are allowed for prediction.")
    try:
        label, probabilities = predict_image(file.file)
        return PredictionResponse(class_label=label, probabilities=probabilities)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

@app.get("/health")
def health():
    return {"status": "ok", "message": "API is healthy and model is loaded."}


