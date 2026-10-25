FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY requirements-agent.txt .
RUN pip install --no-cache-dir -r requirements-agent.txt
RUN useradd --create-home --uid 1000 docspilot \
    && mkdir -p /app/data /app/model_cache \
    && chown -R docspilot:docspilot /app/data /app/model_cache
COPY app ./app
COPY tests ./tests
USER docspilot

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
