FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 200 --retries 5 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.task_a_api:app", "--host", "0.0.0.0", "--port", "8000"]
