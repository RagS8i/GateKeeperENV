FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.lock ./
RUN python -m pip install --no-cache-dir -r requirements.lock

COPY gatekeeper ./gatekeeper
COPY tasks.json ./tasks.json

RUN mkdir -p /app/logs /app/inputs /tmp/gatekeeper

EXPOSE 8000

CMD ["python", "-m", "gatekeeper.harness.run"]
