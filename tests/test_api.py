"""ทดสอบ API ด้วย predictor ปลอม (ไม่ต้องมีโมเดลจริง)"""

import pytest
from fastapi.testclient import TestClient

from app import main
from src.models.classifier import Detection, classify
from tests.conftest import card_image, jpeg_bytes


class FakePredictor:
    def __init__(self, params):
        self.params = params
        self.reference = {}

    def predict(self, img):
        dets = [Detection("Card", 0.9, (0, 0, *img.size)), Detection("Scratch", 0.8, (10, 10, 40, 40))]
        return classify(dets, self.params["classifier"])


@pytest.fixture
def client(tmp_path, monkeypatch, params):
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    main.state.predictor, main.state.version = FakePredictor(params), "1"
    yield TestClient(main.app)
    main.state.predictor, main.state.version = None, None


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_version"] == "1"


def test_predict_ok_and_logged(client, tmp_path):
    r = client.post("/predict", files={"file": ("c.jpg", jpeg_bytes(card_image()), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "defective" and body["n_defects"] == 1 and body["defect_probability"] == 0.8
    assert (tmp_path / "predictions.jsonl").read_text().count("\n") == 1


def test_predict_rejects_bad_input(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422
    assert "cannot_decode" in r.text


def test_batch_mixed(client):
    files = [("files", ("a.jpg", jpeg_bytes(card_image()), "image/jpeg")),
             ("files", ("b.jpg", b"bad", "image/jpeg"))]
    body = client.post("/predict/batch", files=files).json()
    assert body["n"] == 2
    assert "verdict" in body["results"][0] and body["results"][1]["status_code"] == 422


def test_feedback_validates_label(client):
    assert client.post("/feedback", json={"request_id": "x", "true_label": "good"}).status_code == 200
    assert client.post("/feedback", json={"request_id": "x", "true_label": "PSA 10"}).status_code == 422


def test_no_model_returns_503(client):
    main.state.predictor = None
    assert client.get("/ready").status_code == 503
    r = client.post("/predict", files={"file": ("c.jpg", jpeg_bytes(card_image()), "image/jpeg")})
    assert r.status_code == 503


def test_metrics_exposed(client):
    client.post("/predict", files={"file": ("c.jpg", jpeg_bytes(card_image()), "image/jpeg")})
    text = client.get("/metrics").text
    assert "psa_request_latency_seconds" in text and "psa_predicted_verdict_total" in text
