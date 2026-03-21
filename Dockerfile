FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md DECISIONS.md /app/
COPY src /app/src
COPY whisper-omega-plan /app/whisper-omega-plan

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e .

ENTRYPOINT ["omega"]

