# PharmaDemand💊📈

End-to-end ML Engineering project for pharmacy demand forecasting and bundle recommendation.

## Features
- 📦 Demand forecasting (7-day horizon)
- 🛒 Product bundle recommendation
- 🚀 FastAPI inference service
- 🧪 Unit & API tests
- 🐳 Dockerized deployment
- 🔁 GitHub Actions CI

## Tech Stack
- Python, Pandas, Scikit-learn
- FastAPI
- Docker
- GitHub Actions

## API Endpoints
- `GET /health`
- `POST /predict/demand-next7`
- `POST /recommend/bundles`
- `POST /recommend/from-file`

## Run Locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload

