# PharmaDemandOps 💊📈

End-to-end ML Engineering & MLOps project for pharmacy demand forecasting and bundle recommendation.

## Features
- 📦 7-day demand forecasting
- 🛒 Product bundle recommendation (frequently bought together)
- 🚀 FastAPI-based inference service
- 🧪 Automated unit & API tests
- 🐳 Dockerized application
- 🔁 GitHub Actions CI (stable)

## Tech Stack
- Python, Pandas, NumPy, Scikit-learn
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

Run Tests
pytest -q

Run with Docker
docker build -t pharmademandops .
docker run -p 8000:8000 pharmademandops
