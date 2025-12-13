PharmaDemand 💊📈

End-to-end ML Engineering & MLOps project for pharmacy demand forecasting and product bundle recommendation.

Features

📦 7-day demand forecasting

🛒 Product bundle recommendations

🚀 FastAPI inference service

🧪 Automated tests (pytest)

🐳 Dockerized application

🔁 GitHub Actions CI (stable)

Tech Stack

Python, Pandas, NumPy, Scikit-learn

FastAPI

Docker

GitHub Actions

API Endpoints

GET /health

POST /predict/demand-next7

POST /recommend/bundles

POST /recommend/from-file

Run Locally
pip install -r requirements.txt
uvicorn app.main:app --reload


Open API docs at:
👉 http://127.0.0.1:8000/docs


Run Tests
pytest -q

Docker
docker build -t pharmademand .
docker run -p 8000:8000 pharmademand
