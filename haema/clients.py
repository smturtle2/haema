from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str], output_dimensionality: int) -> NDArray[np.float32]:
        """
        Convert each input text into a vector.

        Returns a 2D numpy array with:
        - shape: (len(texts), output_dimensionality)
        - dtype: float32
        """


class LLMClient(ABC):
    @abstractmethod
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
    ) -> dict[str, Any]:
        """
        Generate a structured response for `response_model`.

        Must return a dict that is parseable into `response_model`.
        """
