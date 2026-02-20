from fastapi.testclient import TestClient
from unittest.mock import patch
import io
from PIL import Image
from src.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is healthy and model is loaded."}

@patch('src.main.predict_image')  # ← change this line
def test_predict_success_with_mocked_model(mock_predict):
    mock_predict.return_value = ("cats", [0.85, 0.15])
    dummy_image = Image.new('RGB', (160, 160), color='blue')
    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    response = client.post("/predict", files={"file": ("test.png", img_byte_arr, "image/png")})
    assert response.status_code == 200
    assert response.json()["class_label"] == "cats"
    assert len(response.json()["probabilities"]) == 2

def test_predict_invalid_file_type_handling():
    response = client.post("/predict", files={"file": ("doc.txt", b"not image", "text/plain")})
    assert response.status_code == 400
    assert "Only image files" in response.json()["detail"]

def test_predict_missing_file_upload():
    response = client.post("/predict", data={})
    assert response.status_code == 422