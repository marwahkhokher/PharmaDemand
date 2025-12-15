from pathlib import Path
import json
from fastapi import APIRouter, HTTPException

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_FILE = REPORTS_DIR / "demand_model_metrics.json"

@router.get("/metrics/latest")
def get_latest_metrics():
    if not METRICS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Run training pipeline first."
        )
    return json.loads(METRICS_FILE.read_text(encoding="utf-8-sig"))
