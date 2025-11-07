from fastapi.testclient import TestClient

# adjust import if your path is different
from serving.fastapi_app import app

client = TestClient(app)


def test_healthz_ok():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") is True
    assert "time" in body  # just make sure the field is there


def test_metrics_exposes_prometheus_text():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    # Prometheus text format always has this
    assert b"# HELP" in resp.content
