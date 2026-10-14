# syntax=docker/dockerfile:1

# ------------------------------------------------------------------------------
# Stage 1: Build environment & pre-download model using uv with Python 3.14
# ------------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

ARG MODEL_REPO="litert-community/embeddinggemma-2-740m-litert-lm"
ARG MODEL_FILE="embeddinggemma-2-740m.litertlm"
ARG MODEL_ID="embeddinggemma-2-740m"

WORKDIR /app

ENV LITERT_LM_DIR=/app/litert-lm
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Create isolated Python virtual environment
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install LiteRT-LM CLI and inference API runtime
RUN uv pip install --no-cache litert-lm litert-lm-api

# Pre-download and import model into LiteRT-LM model registry
# Creates both primary model ID and an alias 'embeddinggemma-2'
RUN litert-lm import --from-huggingface-repo="${MODEL_REPO}" "${MODEL_FILE}" "${MODEL_ID}" && \
    mkdir -p /app/litert-lm/models/embeddinggemma-2 && \
    ln /app/litert-lm/models/${MODEL_ID}/model.litertlm /app/litert-lm/models/embeddinggemma-2/model.litertlm

# ------------------------------------------------------------------------------
# Stage 2: Minimal runtime image using Python 3.14 without uv
# ------------------------------------------------------------------------------
FROM python:3.14-slim AS runner

ENV LITERT_LM_DIR=/app/litert-lm
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and pre-downloaded models from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/litert-lm /app/litert-lm

# Configure 8192 context window and 140 vision tokens per frame
COPY sitecustomize.py /app/.venv/lib/python3.14/site-packages/sitecustomize.py

EXPOSE 9379

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:9379/v1/models || exit 1

ENTRYPOINT ["litert-lm", "serve", "--host", "0.0.0.0", "--port", "9379"]
