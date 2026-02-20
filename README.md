# Containerized ML Prediction API

## Description
A production-ready REST API for image classification using a Keras/TensorFlow model, containerized with Docker and deployed via GitHub Actions CI/CD.

## Tech Stack
Python 3.10, FastAPI, TensorFlow/Keras, Docker, GitHub Actions

## Project Structure
...

## Setup
pip install -r requirements.txt

## Run with Docker Compose
docker-compose up --build

## API Usage
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict -F "file=@your_image.jpg"

## Run Tests
pytest tests/

## CI/CD
GitHub Actions workflow triggers on push to main — builds Docker image, runs tests, simulates registry push.

## Predictions Examples
See predictions/ directory for sample JSON outputs.

## Future Enhancements
- Authentication/API keys
- Model versioning
- Monitoring with Prometheus
- Kubernetes deployment