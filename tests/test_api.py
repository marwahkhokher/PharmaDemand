from fastapi.testclient import TestClient
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Import your FastAPI app
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert isinstance(data["feature_columns"], list)
    assert len(data["feature_columns"]) > 0


def test_predict_demand_next7():
    payload = {
        "pharmacy_id": "Ph01_Z01_C01",
        "barcode": "6.25107E+12",
        "date": "2024-12-31",
        "recent_daily_sales": [0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 1, 2, 0, 1],
    }
    r = client.post("/predict/demand-next7", json=payload)
    assert r.status_code == 200
    data = r.json()

    assert data["pharmacy_id"] == payload["pharmacy_id"]
    assert data["barcode"] == payload["barcode"]
    assert "next_day_demand" in data
    assert "next_7day_demand" in data
    assert isinstance(data["next_day_demand"], (int, float))
    assert isinstance(data["next_7day_demand"], (int, float))


def test_recommend_bundles():
    payload = {"basket": ["6.25107E+12", "6.2911E+12"], "top_k": 5}
    r = client.post("/recommend/bundles", json=payload)
    assert r.status_code == 200
    data = r.json()

    assert data["basket"] == payload["basket"]
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    # can be empty depending on data, but should exist and be a list


def test_recommend_from_file_csv_upload():
    # Create a tiny CSV in-memory
    csv_content = "Invoice,barcode\nINV1,6.25107E+12\nINV1,6.2911E+12\nINV2,3.58291E+12\nINV2,6.25107E+12\n"
    files = {"file": ("test_invoices.csv", csv_content, "text/csv")}

    r = client.post("/recommend/from-file?top_k=5", files=files)
    assert r.status_code == 200
    data = r.json()

    assert data["top_k"] == 5
    assert "results" in data
    assert "INV1" in data["results"]
    assert "INV2" in data["results"]
    assert isinstance(data["results"]["INV1"], list)
    assert isinstance(data["results"]["INV2"], list)
