import pytest
from fastapi.testclient import TestClient
import main

VALID = {
    "tenure": 5,
    "MonthlyCharges": 80.5,
    "TotalCharges": 400.0,
    "SeniorCitizen": 0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "TechSupport": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes",
    "Partner": "No",
    "Dependents": "No",
    "MultipleLines": "No",
    "PhoneService": "Yes",
}


@pytest.fixture
def client():
    # Not used as a context manager on purpose: the lifespan (model loading)
    # does not run, so these tests need no trained model file.
    return TestClient(main.app)


def test_unknown_category_is_rejected(client):
    response = client.post("/predict", json={**VALID, "Contract": "Month to month"})
    assert response.status_code == 422


def test_invalid_customer_value_is_rejected(client):
    response = client.post("/predict", json={**VALID, "customer_value": "high"})
    assert response.status_code == 422


def test_missing_total_charges_is_accepted_and_customer_id_is_echoed(client, monkeypatch):
    captured = {}

    def fake_predict(df):
        captured["df"] = df
        return {"churn_probability": 0.5, "recommended_action": "No action"}

    sent = []

    monkeypatch.setattr(main.predictor, "predict_with_explanation", fake_predict)
    monkeypatch.setattr(main, "send_to_n8n", lambda data: sent.append(data) or True)

    payload = {k: v for k, v in VALID.items() if k != "TotalCharges"}
    payload["customer_id"] = "C-1001"

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["customer_id"] == "C-1001"
    assert "customer_id" not in captured["df"].columns
    assert sent and sent[0]["customer_id"] == "C-1001"