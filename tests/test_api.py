from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_predict():
    sample_input = {
        "features": [8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]
    }
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()
