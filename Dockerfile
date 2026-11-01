#python base image
FROM python:3.12-slim  
#setting non secrets default
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2
#makes app the working directory
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
# copy dependecy list into the build
COPY requirements-agent.txt .
RUN pip install --no-cache-dir -r requirements-agent.txt
RUN useradd --create-home --uid 1000 docspilot \
    && mkdir -p /app/data /app/model_cache /app/conversations \
    && chown -R docspilot:docspilot /app/data /app/model_cache /app/conversations
#copies our application code into the image
COPY app ./app
#inlcude tests so our ci can execute them iside image
COPY tests ./tests
USER docspilot

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
