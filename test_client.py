#!/usr/bin/env python3
"""Test script for LiteRT-LM EmbeddingGemma 2 Embedding API."""

from __future__ import annotations

import base64
import json
import math
import sys
import urllib.request

SERVER_URL = "http://localhost:9379"

# 1x1 white JPEG image in base64
SAMPLE_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP///////////////////////////////////"
    "///////////////////////////////////////////////////wgALCAABAAEBAREA"
    "/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def make_request(path: str, data: dict | None = None) -> dict:
    url = f"{SERVER_URL}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def test_models():
    print("Testing GET /v1/models ...")
    res = make_request("/v1/models")
    models = [m["id"] for m in res.get("data", [])]
    print(f"  Available models: {models}")
    assert len(models) > 0, "No models found!"
    return models[0]


def test_text_embedding(model_id: str):
    print("\nTesting POST /v1/embeddings (Text) ...")
    payload = {
        "model": model_id,
        "input": "task: search result | text: LiteRT-LM is an on-device LLM runtime.",
    }
    res = make_request("/v1/embeddings", payload)
    data = res.get("data", [])
    assert len(data) == 1, "Expected 1 embedding result"
    embedding = data[0]["embedding"]
    dim = len(embedding)
    print(f"  Received text embedding: dimension = {dim}")
    print(f"  Sample values: {[round(x, 4) for x in embedding[:5]]}")
    assert dim == 768, f"Expected 768 dimensions, got {dim}"
    return embedding


def test_image_embedding(model_id: str):
    print("\nTesting POST /v1/embeddings (Image via Base64 URL) ...")
    payload = {
        "model": model_id,
        "input": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{SAMPLE_JPEG_B64}"
                },
            }
        ],
    }
    res = make_request("/v1/embeddings", payload)
    data = res.get("data", [])
    assert len(data) == 1, "Expected 1 embedding result"
    embedding = data[0]["embedding"]
    dim = len(embedding)
    print(f"  Received image embedding: dimension = {dim}")
    print(f"  Sample values: {[round(x, 4) for x in embedding[:5]]}")
    assert dim == 768, f"Expected 768 dimensions, got {dim}"
    return embedding


def test_dimension_truncation(model_id: str):
    print("\nTesting POST /v1/embeddings (Matryoshka Truncation: 256d) ...")
    payload = {
        "model": model_id,
        "input": "task: search query | text: Retrieval Augmented Generation",
        "dimensions": 256,
    }
    res = make_request("/v1/embeddings", payload)
    embedding = res["data"][0]["embedding"]
    dim = len(embedding)
    print(f"  Received truncated embedding: dimension = {dim}")
    assert dim == 256, f"Expected 256 dimensions, got {dim}"


def main():
    print("=" * 60)
    print(" LiteRT-LM EmbeddingGemma 2 Automated Test Suite")
    print("=" * 60)
    try:
        model_id = test_models()
        text_emb = test_text_embedding(model_id)
        img_emb = test_image_embedding(model_id)
        test_dimension_truncation(model_id)

        sim = cosine_similarity(text_emb, img_emb)
        print(f"\nCosine similarity between text and image embeddings: {sim:.4f}")
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
