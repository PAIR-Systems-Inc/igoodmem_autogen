"""Configuration classes for GoodMem vector memory."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """Configuration for how documents are chunked before embedding.

    Mirrors the reference integration's advanced chunking options.
    """

    chunk_size: int = Field(default=256, description="Number of characters per chunk when splitting documents")
    chunk_overlap: int = Field(default=25, description="Number of overlapping characters between consecutive chunks")
    keep_strategy: Literal["KEEP_END", "KEEP_START", "DISCARD"] = Field(
        default="KEEP_END", description="Where to attach the separator when splitting"
    )
    length_measurement: Literal["CHARACTER_COUNT", "TOKEN_COUNT"] = Field(
        default="CHARACTER_COUNT", description="How chunk size is measured"
    )
    separators: List[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""], description="Separators used for recursive splitting"
    )
    separator_is_regex: bool = Field(default=False, description="Whether separators are regex patterns")


class PostProcessorConfig(BaseModel):
    """Configuration for retrieval post-processing (reranker and/or LLM).

    Mirrors the reference integration's reranker/LLM options.
    """

    reranker_id: Optional[str] = Field(default=None, description="Optional reranker model ID to improve result ordering")
    llm_id: Optional[str] = Field(default=None, description="Optional LLM ID to generate contextual responses")
    relevance_threshold: Optional[float] = Field(
        default=None, description="Minimum score (0-1) for including results"
    )
    llm_temperature: Optional[float] = Field(
        default=None, description="Creativity setting for LLM generation (0-2)"
    )
    chronological_resort: bool = Field(
        default=False, description="Reorder results by creation time instead of relevance score"
    )


class GoodMemMemoryConfig(BaseModel):
    """Configuration for GoodMem memory implementation.

    This configuration connects to a GoodMem API server for vector-based memory
    storage and semantic retrieval. It mirrors the functionality of the reference
    Activepieces GoodMem integration, adapted for AutoGen's Python ecosystem.

    Args:
        base_url: The base URL of your GoodMem API server.
        api_key: Your GoodMem API key for authentication (X-API-Key).
        space_name: Name of the space to use. If it already exists, it will be reused.
        embedder_id: The embedder model ID for converting text to vector embeddings.
            Use :meth:`GoodMemMemory.list_embedders` to discover available embedders.
        max_results: Maximum number of results to return when querying.
        include_memory_definition: Whether to fetch full memory metadata alongside matched chunks.
        wait_for_indexing: Retry for up to 60 seconds when no results are found during retrieval.
        chunking: Advanced chunking configuration for document splitting.
        post_processor: Optional post-processing configuration for reranking and LLM generation.

    Example:
        .. code-block:: python

            from autogen_ext.memory.goodmem import GoodMemMemory, GoodMemMemoryConfig

            config = GoodMemMemoryConfig(
                base_url="http://localhost:8080",
                api_key="gm_your_api_key",
                space_name="my-knowledge-base",
                embedder_id="your-embedder-id",
            )
            memory = GoodMemMemory(config=config)
    """

    base_url: str = Field(description="The base URL of your GoodMem API server")
    api_key: str = Field(description="Your GoodMem API key for authentication (X-API-Key)")
    space_name: str = Field(description="Name of the space to use. Created if it does not exist, reused if it does")
    embedder_id: str = Field(description="The embedder model ID for vector embeddings")
    max_results: int = Field(default=5, description="Maximum number of results to return when querying")
    include_memory_definition: bool = Field(
        default=True,
        description="Fetch the full memory metadata alongside matched chunks during retrieval",
    )
    wait_for_indexing: bool = Field(
        default=True,
        description="Retry for up to 60 seconds when no results are found during retrieval",
    )
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig, description="Advanced chunking configuration")
    post_processor: Optional[PostProcessorConfig] = Field(
        default=None, description="Optional post-processing configuration for reranking and LLM generation"
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None, description="Default metadata to attach to all memories created via add()"
    )
    verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates. Set to False for self-signed certificates.",
    )
