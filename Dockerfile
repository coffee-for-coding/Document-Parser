FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download de_core_news_sm && \
    python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000
CMD ["uvicorn", "docsearch.server:app", "--host", "0.0.0.0", "--port", "8000"]
