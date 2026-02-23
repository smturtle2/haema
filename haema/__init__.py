"""Top-level public API for HAEMA.

Import stable user-facing contracts directly from `haema`:

Example:
    from haema import Memory, EmbeddingClient, LLMClient
"""

from haema.clients import EmbeddingClient, LLMClient
from haema.memory import Memory

# Only stable user-facing symbols are re-exported from the package root.
__all__ = ["EmbeddingClient", "LLMClient", "Memory"]
