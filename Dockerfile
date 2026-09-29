FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY artifacts ./artifacts
COPY config ./config

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "telco_nba.api:app", "--host", "0.0.0.0", "--port", "8000"]
