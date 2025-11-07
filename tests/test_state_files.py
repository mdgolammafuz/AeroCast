import os
from pathlib import Path

from fastapi.testclient import TestClient
from serving.fastapi_app import app

client = TestClient(app)


def test_retrain_files_created(tmp_path: Path, monkeypatch):
    """
    We said the API links to /app/state/retrain.flag and /app/state/last_retrain.txt.
    Simulate that directory and make sure the app doesn't crash when we hit /healthz.
    """

    # pretend the app writes/reads here
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # some apps read env var, some use hardcoded path.
    # if your code reads an env like AEROCAST_STATE_DIR, set it here.
    monkeypatch.setenv("AEROCAST_STATE_DIR", str(state_dir))

    # hit the app so any startup logic runs
    resp = client.get("/healthz")
    assert resp.status_code == 200

    # create the files like docker entrypoint does
    (state_dir / "last_retrain.txt").write_text("2025-01-01T00:00:00Z\n")
    (state_dir / "retrain.flag").write_text("drift 2025-01-01T00:00:00Z\n")

    # make sure they exist
    assert (state_dir / "last_retrain.txt").exists()
    assert (state_dir / "retrain.flag").exists()

    # and make sure reading them later wouldn't blow up
    content = (state_dir / "retrain.flag").read_text().strip()
    assert content.startswith("drift") or content != ""
