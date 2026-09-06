FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=0.0.0.0:11434
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /.ollama && chmod 777 /.ollama
ENV OLLAMA_MODELS=/.ollama/models

COPY . .

EXPOSE 7860

CMD ollama serve & sleep 5 && uvicorn main:app --host 0.0.0.0 --port 7860