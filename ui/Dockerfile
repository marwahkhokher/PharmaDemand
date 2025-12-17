FROM python:3.11-slim

WORKDIR /app

# Install dependencies (using the same requirements.txt you already have)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project code (so ui/app.py exists in container)
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
