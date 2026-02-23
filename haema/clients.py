from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class EmbeddingClient(ABC):
    """Abstract embedding adapter used by :class:`haema.Memory`.

    Implementations must provide separate query/document embedding methods.
    This allows providers to apply different task settings for retrieval queries
    and stored memory documents.
    """

    @abstractmethod
    def embed_query(self, texts: Sequence[str], output_dimensionality: int) -> NDArray[np.float32]:
        """Embed retrieval query texts.

        Args:
            texts: Query strings to embed as one batch.
            output_dimensionality: Expected embedding width for each query vector.

        Returns:
            A 2D `numpy.ndarray` with dtype `float32` and shape
            `(len(texts), output_dimensionality)`.

        Raises:
            Exception: Provider-specific errors should be propagated.
        """

    @abstractmethod
    def embed_document(self, texts: Sequence[str], output_dimensionality: int) -> NDArray[np.float32]:
        """Embed memory document texts for storage.

        Args:
            texts: Document strings to embed as one batch.
            output_dimensionality: Expected embedding width for each document vector.

        Returns:
            A 2D `numpy.ndarray` with dtype `float32` and shape
            `(len(texts), output_dimensionality)`.

        Raises:
            Exception: Provider-specific errors should be propagated.
        """


class LLMClient(ABC):
    """Abstract structured-output LLM adapter used by :class:`haema.Memory`."""

    @abstractmethod
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
    ) -> dict[str, Any]:
        """Generate a structured response for the requested schema.

        Args:
            system_prompt: System-level instruction text.
            user_prompt: User-level prompt text for the current task.
            response_model: Pydantic model class used as the output schema.

        Returns:
            A dictionary parseable by `response_model.model_validate(...)`.

        Raises:
            Exception: Provider-specific errors should be propagated.

        Example:
            class MyModel(BaseModel):
                value: str

            raw = client.generate_structured("sys", "user", MyModel)
            parsed = MyModel.model_validate(raw)
        """
