import pytest
from fastapi.testclient import TestClient

from conftest import CLASSES, image_bytes
from foodvision.api import MAX_UPLOAD_BYTES, create_app


@pytest.fixture(scope="module")
def client(artifacts_dir):
    with TestClient(create_app(artifacts_dir)) as c:
        yield c


def post_image(client, data, name="dish.jpg", ctype="image/jpeg", **params):
    return client.post("/predict", files={"file": (name, data, ctype)}, params=params)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["num_classes"] == len(CLASSES)


def test_classes(client):
    assert client.get("/classes").json()["classes"] == CLASSES


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Food Vision" in r.text


def test_predict_returns_top3_with_nutrition(client):
    r = post_image(client, image_bytes())
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3
    for p in body["predictions"]:
        assert set(p) >= {"dish", "name", "confidence", "nutrition"}
        assert p["nutrition"]["kcal"] > 0
    assert isinstance(body["confident"], bool)
    assert body["latency_ms"] >= 0
    assert "Not dietary advice" in body["nutrition_disclaimer"]


def test_low_confidence_message(client):
    # Random weights on 3 classes -> top confidence is near 1/3, below the threshold
    body = post_image(client, image_bytes(seed=11)).json()
    if not body["confident"]:
        assert "20 Indian dishes" in body["message"]


def test_png_upload(client):
    assert post_image(client, image_bytes(fmt="PNG"), "x.png", "image/png").status_code == 200


def test_top_k_param(client):
    assert len(post_image(client, image_bytes(), top_k=1).json()["predictions"]) == 1
    assert post_image(client, image_bytes(), top_k=0).status_code == 422


def test_rejects_non_image(client):
    r = post_image(client, b"definitely not an image", "notes.txt", "text/plain")
    assert r.status_code == 400


def test_rejects_empty_file(client):
    assert post_image(client, b"").status_code == 400


def test_rejects_oversized_file(client):
    assert post_image(client, b"0" * (MAX_UPLOAD_BYTES + 10)).status_code == 413


def test_missing_artifacts_fails_fast(tmp_path):
    with pytest.raises(RuntimeError, match="model.pt not found"):
        with TestClient(create_app(tmp_path)):
            pass
