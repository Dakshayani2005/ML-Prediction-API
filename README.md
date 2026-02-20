# 🤖 Containerized ML Prediction API

A production-ready RESTful API for image classification using a Keras/TensorFlow deep learning model, containerized with Docker and automated via a GitHub Actions CI/CD pipeline.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Local Development with Docker Compose](#local-development-with-docker-compose)
- [API Usage Examples](#api-usage-examples)
- [Running Tests](#running-tests)
- [CI/CD Pipeline](#cicd-pipeline)
- [Prediction Examples](#prediction-examples)
- [Future Enhancements](#future-enhancements)

---

## 📌 Project Overview

This project transforms a trained Keras image classification model into a fully containerized, production-grade REST API. The model classifies images into one of two categories — **cats** or **dogs** — and returns **"unknown"** when it is not sufficiently confident in its prediction (e.g., when given an image of a frog or an unrelated object).

The project demonstrates core MLOps principles including:
- Model serving via a FastAPI REST interface
- Docker containerization with multi-stage builds
- Automated CI/CD using GitHub Actions
- Structured logging, input validation, and robust error handling

---

## ✨ Features

- `POST /predict` endpoint accepting image file uploads and returning classification results
- `GET /health` endpoint for liveness monitoring
- Confidence thresholding — returns `"unknown"` for low-confidence predictions
- Model loaded once on application startup for low-latency inference
- Input validation with appropriate HTTP status codes (400, 422, 500)
- Multi-stage Docker build for optimized image size
- Docker Compose for easy local development
- GitHub Actions CI/CD pipeline (build, test, Docker image build, simulated push)
- Auto-generated Swagger/OpenAPI documentation at `/docs`
- Structured logging for all requests and prediction events
- All configuration via environment variables

---

## 🛠 Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| API Framework | FastAPI |
| ASGI Server | Uvicorn |
| Deep Learning | TensorFlow 2.x / Keras |
| Model Architecture | MobileNetV2 (transfer learning) |
| Image Processing | Pillow (PIL) |
| Data Handling | NumPy |
| Validation | Pydantic v2 |
| Testing | Pytest |
| Containerization | Docker (multi-stage build) |
| Orchestration | Docker Compose |
| CI/CD | GitHub Actions |

---

## 📁 Project Structure

```
ML Prediction API/
├── .github/
│   └── workflows/
│       └── main.yml          # GitHub Actions CI/CD workflow
├── src/
│   ├── __init__.py
│   ├── main.py               # FastAPI app, routes, lifespan
│   ├── model.py              # Model loading, preprocessing, inference
│   ├── schemas.py            # Pydantic response models
│   └── config.py             # Settings via pydantic-settings
├── models/
│   └── my_classifier_model.h5  # Trained Keras model
├── tests/
│   └── test_api.py           # Pytest unit tests
├── predictions/
│   └── example_prediction.json # Sample API output
├── .env.example              # Environment variable reference
├── conftest.py               # Pytest path configuration
├── pytest.ini                # Pytest settings
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # Local development setup
├── requirements.txt          # Python dependencies
├── train_model.py            # Model training script
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ml-prediction-api.git
cd ml-prediction-api
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` if needed:
```
MODEL_PATH=models/my_classifier_model.h5
LOG_LEVEL=INFO
```

### 5. Ensure the Model File Exists

The trained model should be at `models/my_classifier_model.h5`. If you need to train it from scratch:

```bash
python train_model.py
```

> **Note:** Training requires a dataset in `database/train` and `database/test` directories, organized into `cats/` and `dogs/` subdirectories.

### 6. Run the API Locally

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## 🐳 Local Development with Docker Compose

Build and start the service with a single command:

```bash
docker-compose up --build
```

This will:
1. Build the Docker image using the multi-stage Dockerfile
2. Mount the `models/` directory into the container
3. Start the API on port 8000
4. Run a health check every 10 seconds

To stop the service:

```bash
docker-compose down
```

To check the container health status:

```bash
docker ps
```

Look for `(healthy)` in the STATUS column after about 30–60 seconds.

---

## 🔌 API Usage Examples

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "message": "API is healthy and model is loaded."
}
```

---

### Predict — Valid Image (Cat or Dog)

```bash
curl -X POST http://localhost:8000/predict \
  -H "accept: application/json" \
  -F "file=@/path/to/your/cat.jpg;type=image/jpeg"
```

**Response (200 OK):**
```json
{
  "class_label": "cats",
  "probabilities": [0.9231, 0.0769]
}
```

---

### Predict — Unknown/Out-of-Distribution Image (e.g., frog)

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/frog.jpg;type=image/jpeg"
```

**Response (200 OK):**
```json
{
  "class_label": "unknown",
  "probabilities": [0.5312, 0.4688]
}
```

> The model returns `"unknown"` when its confidence is below the threshold (default: 85%).

---

### Predict — Invalid File Type

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@document.txt;type=text/plain"
```

**Response (400 Bad Request):**
```json
{
  "detail": "Only image files (e.g., JPEG, PNG) are allowed for prediction."
}
```

---

### Predict — Missing File

```bash
curl -X POST http://localhost:8000/predict
```

**Response (422 Unprocessable Entity):**
```json
{
  "detail": [{ "msg": "field required", "type": "missing" }]
}
```

---

### Swagger UI

For interactive testing, visit:
```
http://localhost:8000/docs
```

---

## 🧪 Running Tests

Run the full test suite with:

```bash
pytest tests/ -v
```

Expected output:

```
tests/test_api.py::test_health_check_endpoint              PASSED
tests/test_api.py::test_predict_success_with_mocked_model  PASSED
tests/test_api.py::test_predict_invalid_file_type_handling PASSED
tests/test_api.py::test_predict_missing_file_upload        PASSED

4 passed in Xs
```

**What's tested:**
- Health endpoint returns correct status and message
- Successful prediction with a mocked model (isolated, fast, no real inference)
- Invalid file type returns 400 with correct error message
- Missing file returns 422 Unprocessable Entity

---

## 🔄 CI/CD Pipeline

The CI/CD pipeline is defined in `.github/workflows/main.yml` and uses **GitHub Actions**.

### Trigger

The pipeline runs automatically on:
- Every `push` to the `main` branch
- Every `pull_request` targeting `main`

### Pipeline Steps

| Step | Description |
|---|---|
| Checkout code | Pulls the latest repository content |
| Set up Python 3.10 | Configures the runner environment |
| Install dependencies | Runs `pip install -r requirements.txt` |
| Run Pytest | Executes all unit tests in `tests/` |
| Build Docker image | Builds and tags with Git commit SHA and `latest` |
| Login to registry | Placeholder (skipped for demo) |
| Push Docker image | Placeholder (skipped for demo) |
| Save prediction examples | Writes example JSON outputs to `predictions/` |
| Upload artifacts | Uploads `predictions/` as a downloadable workflow artifact |

### Viewing Pipeline Status

1. Go to your GitHub repository
2. Click the **Actions** tab
3. Click the latest workflow run to see detailed logs for each step

### Adding Real Registry Push (Optional)

To push to Docker Hub in production, add these secrets to your GitHub repository settings:
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

Then replace the placeholder steps in `main.yml` with:

```yaml
- name: Log in to Docker Hub
  uses: docker/login-action@v2
  with:
    username: ${{ secrets.DOCKER_USERNAME }}
    password: ${{ secrets.DOCKER_PASSWORD }}

- name: Push Docker image
  run: docker push your-dockerhub-username/my-ml-api:${{ github.sha }}
```

---

## 📂 Prediction Examples

The `predictions/` directory contains sample JSON outputs from successful `/predict` calls:

**`predictions/example_prediction.json`**
```json
{
  "class_label": "dogs",
  "probabilities": [0.08, 0.92]
}
```

- `class_label`: The predicted class (`"cats"`, `"dogs"`, or `"unknown"`)
- `probabilities`: A list of `[P(cats), P(dogs)]` summing to approximately 1.0

---

## 🔧 Environment Variables

All configurable parameters are managed via environment variables. See `.env.example`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/my_classifier_model.h5` | Path to the trained Keras model |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 🚀 Future Enhancements

- **Authentication** — Add API key or OAuth2 authentication to secure the `/predict` endpoint
- **Multi-class support** — Extend the model to classify more than two categories
- **Model versioning** — Support loading different model versions via environment config
- **Monitoring** — Integrate Prometheus metrics and Grafana dashboards for request tracking
- **Rate limiting** — Prevent API abuse with request throttling
- **Async inference** — Use background tasks for heavy inference workloads
- **Kubernetes deployment** — Define Helm charts for scalable cloud deployment
- **Model drift detection** — Monitor prediction distributions over time to catch data drift
- **GPU support** — Configure Docker and TensorFlow for GPU-accelerated inference
- **Integration tests** — Add end-to-end tests that spin up the full Docker container

---

