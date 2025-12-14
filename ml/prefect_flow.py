from __future__ import annotations

import os
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys


from prefect import flow, task, get_run_logger

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Scripts you already have
INGEST_SCRIPT = ML_DIR / "ingest_kaggle_pharmacy.py"
TRAIN_SCRIPT = ML_DIR / "train.py"
QA_SCRIPT = ML_DIR / "qa_checks.py"  # we'll use this as your "ML testing gate" for now

# Artifacts your pipeline should produce
MODEL_FILE = MODELS_DIR / "demand_regressor.pkl"
METRICS_FILE = REPORTS_DIR / "demand_model_metrics.json"


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    """Run a command and raise if it fails."""
    subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT), check=True)


@task(retries=2, retry_delay_seconds=10)
def ingest_data() -> None:
    logger = get_run_logger()
    logger.info("Running ingestion script...")
    if not INGEST_SCRIPT.exists():
        raise FileNotFoundError(f"Missing: {INGEST_SCRIPT}")
    _run([sys.executable, str(INGEST_SCRIPT)])
    logger.info("Ingestion complete.")


@task(retries=2, retry_delay_seconds=10)
def train_model() -> None:
    logger = get_run_logger()
    logger.info("Running training script...")
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing: {TRAIN_SCRIPT}")
    _run([sys.executable, str(TRAIN_SCRIPT)])
    logger.info("Training complete.")


@task(retries=1, retry_delay_seconds=5)
def ml_quality_gate() -> None:
    """
    This is your automated ML testing gate.
    For now it runs ml/qa_checks.py (schema + sanity checks).
    In Step 2 we'll add DeepChecks and call it here as well.
    """
    logger = get_run_logger()
    logger.info("Running ML quality gate checks...")
    if not QA_SCRIPT.exists():
        raise FileNotFoundError(f"Missing: {QA_SCRIPT}")
    _run([sys.executable, str(QA_SCRIPT)])
    logger.info("Quality gate passed.")


@task
def version_artifacts() -> Path:
    """
    Save and version artifacts under:
      models/versions/<timestamp>/
    and also refresh models/latest/ for runtime use.
    """
    logger = get_run_logger()

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    versions_dir = MODELS_DIR / "versions" / ts
    latest_dir = MODELS_DIR / "latest"

    versions_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model artifact missing: {MODEL_FILE}")
    if not METRICS_FILE.exists():
        # metrics is optional for CI-safe runtime, but for pipeline we expect it after training
        raise FileNotFoundError(f"Metrics artifact missing: {METRICS_FILE}")

    # Copy into timestamped version
    shutil.copy2(MODEL_FILE, versions_dir / MODEL_FILE.name)
    shutil.copy2(METRICS_FILE, versions_dir / METRICS_FILE.name)

    # Refresh latest/
    shutil.copy2(MODEL_FILE, latest_dir / MODEL_FILE.name)
    shutil.copy2(METRICS_FILE, latest_dir / METRICS_FILE.name)

    logger.info(f"Versioned artifacts at: {versions_dir}")
    logger.info(f"Updated latest artifacts at: {latest_dir}")

    return versions_dir


@task
def notify(status: str, message: str) -> None:
    """
    Optional notifications via Apprise.
    If you set APPRISE_URL, it will send notifications.
    Otherwise it just logs.
    """
    logger = get_run_logger()
    apprise_url = os.getenv("APPRISE_URL", "").strip()

    logger.info(f"[NOTIFY] {status}: {message}")

    if not apprise_url:
        logger.info("APPRISE_URL not set. Skipping external notification.")
        return

    try:
        import apprise  # already in your requirements.txt
        a = apprise.Apprise()
        a.add(apprise_url)
        a.notify(title=f"PharmaDemandOps: {status}", body=message)
        logger.info("Notification sent via Apprise.")
    except Exception as e:
        logger.warning(f"Failed to send notification: {e}")


@flow(name="pharmademandops_pipeline")
def pharmademandops_pipeline(run_ingest: bool = True) -> Path:
    """
    Full ML lifecycle flow:
      - ingest (optional)
      - train
      - ML tests gate
      - version/save artifacts
      - notify success/failure
    """
    logger = get_run_logger()

    try:
        if run_ingest:
            ingest_data()
        train_model()
        ml_quality_gate()
        version_dir = version_artifacts()
        notify("SUCCESS", f"Pipeline completed. Artifacts saved to {version_dir}")
        return version_dir

    except Exception as e:
        logger.exception("Pipeline failed.")
        notify("FAILED", str(e))
        raise


if __name__ == "__main__":
    # Local run
    pharmademandops_pipeline(run_ingest=True)
