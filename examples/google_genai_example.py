from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from haema import EmbeddingClient, LLMClient, Memory

STORAGE_DIR = Path("./example_haema")
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3-flash-preview"
OUTPUT_DIMENSIONALITY = 128


class GoogleGenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, client: genai.Client, model: str = "gemini-embedding-001") -> None:
        self.client = client
        self.model = model

    def embed(self, texts: list[str], output_dimensionality: int) -> np.ndarray:
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
        )
        vectors = [item.values for item in (response.embeddings or [])]
        return np.asarray(vectors, dtype=np.float32)


class GoogleGenAILLMClient(LLMClient):
    def __init__(self, client: genai.Client, model: str = "gemini-3-flash-preview") -> None:
        self.client = client
        self.model = model

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=response_model.model_json_schema(),
            ),
        )

        if isinstance(response.parsed, dict):
            return response.parsed
        if isinstance(response.text, str) and response.text.strip():
            return json.loads(response.text)
        raise ValueError("LLM response did not contain parseable JSON")


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing GOOGLE_API_KEY environment variable.")

    client = genai.Client(api_key=api_key, vertexai=False)
    embedding_client = GoogleGenAIEmbeddingClient(client=client, model=EMBEDDING_MODEL)
    llm_client = GoogleGenAILLMClient(client=client, model=LLM_MODEL)

    memory = Memory(
        path=STORAGE_DIR,
        output_dimensionality=OUTPUT_DIMENSIONALITY,
        embedding_client=embedding_client,
        llm_client=llm_client,
    )

    memory.add(
        [
            "User prefers concise, actionable answers.",
            "User works on HAEMA memory framework using ChromaDB.",
            "User asked to test google-genai with Gemini flash and embedding models.",
        ]
    )

    print("=== CORE ===")
    print(memory.get_core())
    print("=== LATEST(1,3) ===")
    print(memory.get_latest(begin=1, count=3))
    print("=== SEARCH ===")
    print(memory.search("What model does the user want?", n=3))


if __name__ == "__main__":
    main()
