# LiteRT-LM Embedding Server (EmbeddingGemma 2 740M)

A Docker image providing an OpenAI-compatible HTTP API (`/v1/embeddings`) for multimodal text and image embeddings powered by Google AI Edge's **LiteRT-LM** and **EmbeddingGemma 2 740M**.

The model is pre-downloaded and registered during the Docker build, enabling completely offline and network-isolated execution.

---

## Features

- **Omnimodal Embedding Backbone**: Powered by `EmbeddingGemma 2 740M` with unified 768-dimensional vector space across text and image.
- **OpenAI-Compatible API**: Implements standard `/v1/models` and `/v1/embeddings` endpoints.
- **Multimodal Inputs**: Supports text prompts and image base64 data URIs (`data:image/jpeg;base64,...`).
- **Offline Capable**: Model package is pre-baked into the image; runs with `--network none`.
- **Matryoshka Representation Learning (MRL)**: Supports custom dimension truncation (e.g. `dimensions: 256`, `512`, `768`).
- **Lightweight & Fast**: Built with `uv` in a multi-stage Docker build, producing a minimal Python 3.14 Debian slim runtime.

---

## Quickstart

### 1. Build Image

```bash
docker build -t litert-lm-embeddinggemma2:740m .
```

*(Optional build args: `MODEL_REPO`, `MODEL_FILE`, `MODEL_ID` to use other variants such as `embeddinggemma-2-text-vision-440m-litert-lm`).*

### 2. Run Container

Using Docker CLI:
```bash
docker run -d \
  --name litert-embedding \
  -p 9379:9379 \
  litert-lm-embeddinggemma2:740m
```

Or using Docker Compose:
```bash
docker compose up -d
```

To run in **completely offline / air-gapped mode**:
```bash
docker run -d \
  --name litert-embedding-offline \
  --network none \
  -p 9379:9379 \
  litert-lm-embeddinggemma2:740m
```

---

## API Usage Examples

The server runs by default on port `9379`.

### 1. Check Available Models

```bash
curl http://localhost:9379/v1/models
```

Output:
```json
{
  "object": "list",
  "data": [
    {
      "id": "embeddinggemma-2-740m",
      "object": "model",
      "owned_by": "litert-lm"
    },
    {
      "id": "embeddinggemma-2",
      "object": "model",
      "owned_by": "litert-lm"
    }
  ]
}
```

### 2. Generate Text Embedding

```bash
curl -X POST http://localhost:9379/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embeddinggemma-2-740m",
    "input": "task: search result | text: LiteRT-LM enables on-device LLM inference."
  }'
```

### 3. Generate Image Embedding (Base64)

Pass images formatted as standard OpenAI multimodal image content parts:

```bash
curl -X POST http://localhost:9379/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embeddinggemma-2-740m",
    "input": [
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
        }
      }
    ]
  }'
```

### 4. Truncate Dimensions (Matryoshka MRL)

```bash
curl -X POST http://localhost:9379/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embeddinggemma-2-740m",
    "input": "task: search query | text: semantic search",
    "dimensions": 256
  }'
```

---

## Automated Testing

Run the included test suite against the running container:

```bash
python3 test_client.py
```
