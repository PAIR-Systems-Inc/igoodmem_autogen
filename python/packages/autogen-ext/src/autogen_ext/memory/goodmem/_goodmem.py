""" vector memory implementation for AutoGen.

Provides semantic memory storage and retrieval using the  API,
supporting text and file-based (PDF, DOCX, images, etc.) memory creation
with vector embeddings.
"""

import asyncio
import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from typing_extensions import Self

from ._goodmem_configs import GoodMemMemoryConfig

logger = logging.getLogger(__name__)


class GoodMemMemory(Memory, Component[GoodMemMemoryConfig]):
    """Store and retrieve memory using the  API with vector similarity search.

    ``Memory`` provides a cloud/self-hosted memory implementation that uses
    the  API for storing documents (text and files) as memories with vector
    embeddings and performing semantic similarity-based retrieval.

    This implementation mirrors the reference Activepieces  integration,
    adapted idiomatically for AutoGen's Python ecosystem. It supports:

    - **Create Space** — Create or reuse a named space with a configured embedder
    - **Create Memory** — Store text or files (PDF, DOCX, images) as memories
    - **Retrieve Memories** — Semantic search across spaces with optional reranking
    - **Get Memory** — Fetch a specific memory by ID with optional content
    - **Delete Memory** — Permanently remove a memory and its embeddings
    - **List Embedders** — Discover available embedding models
    - **List Spaces** — List all available spaces

    Install the required dependency:

    .. code-block:: bash

        pip install "autogen-ext[]"

    Args:
        config: Configuration for the  memory connection.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_agentchat.agents import AssistantAgent
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory. import Memory, MemoryConfig
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            async def main() -> None:
                memory = Memory(
                    config=MemoryConfig(
                        base_url="http://localhost:8080",
                        api_key="gm_your_api_key",
                        space_name="my-knowledge-base",
                        embedder_id="your-embedder-id",
                    )
                )

                # Add text memory
                await memory.add(
                    MemoryContent(
                        content="The capital of France is Paris",
                        mime_type=MemoryMimeType.TEXT,
                    )
                )

                # Add a PDF file as memory
                await memory.add_file("/path/to/document.pdf")

                # Query memories
                results = await memory.query("What is the capital of France?")
                for result in results.results:
                    print(result.content)

                # Use with an agent
                assistant = AssistantAgent(
                    name="assistant",
                    model_client=OpenAIChatCompletionClient(model="gpt-4.1"),
                    memory=[memory],
                )

                await memory.close()

            asyncio.run(main())
    """

    component_config_schema = GoodMemMemoryConfig
    component_provider_override = "autogen_ext.memory.goodmem.GoodMemMemory"

    def __init__(self, config: GoodMemMemoryConfig) -> None:
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._headers = {
            "X-API-Key": config.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._client: Optional[httpx.AsyncClient] = None
        self._space_id: Optional[str] = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure the HTTP client is initialized."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0, verify=self._config.verify_ssl)
        return self._client

    async def _ensure_space(self) -> str:
        """Ensure the space exists, creating it if necessary. Returns the space ID.

        Mirrors the reference integration's create-space logic: checks for an existing
        space with the same name before creating a new one.
        """
        if self._space_id is not None:
            return self._space_id

        client = self._ensure_client()

        # Check if a space with the same name already exists
        try:
            response = await client.get(
                f"{self._base_url}/v1/spaces",
                headers=self._headers,
            )
            response.raise_for_status()
            body = response.json()
            spaces = body if isinstance(body, list) else body.get("spaces", [])

            for space in spaces:
                if space.get("name") == self._config.space_name:
                    self._space_id = space.get("spaceId") or space.get("id")
                    logger.info(f"Reusing existing  space: {self._config.space_name} ({self._space_id})")
                    return self._space_id  # type: ignore[return-value]
        except Exception:
            logger.debug("Failed to list spaces, will attempt to create")

        # Create new space with chunking config
        chunking = self._config.chunking
        request_body: Dict[str, Any] = {
            "name": self._config.space_name,
            "spaceEmbedders": [{"embedderId": self._config.embedder_id, "defaultRetrievalWeight": 1.0}],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunking.chunk_size,
                    "chunkOverlap": chunking.chunk_overlap,
                    "separators": chunking.separators,
                    "keepStrategy": chunking.keep_strategy,
                    "separatorIsRegex": chunking.separator_is_regex,
                    "lengthMeasurement": chunking.length_measurement,
                },
            },
        }

        response = await client.post(
            f"{self._base_url}/v1/spaces",
            headers=self._headers,
            json=request_body,
        )

        # Handle 409 Conflict: space was created between our list check and create
        if response.status_code == 409:
            logger.debug("Space creation returned 409 Conflict, re-listing to find existing space")
            list_response = await client.get(
                f"{self._base_url}/v1/spaces",
                headers=self._headers,
            )
            list_response.raise_for_status()
            body = list_response.json()
            spaces = body if isinstance(body, list) else body.get("spaces", [])
            for space in spaces:
                if space.get("name") == self._config.space_name:
                    self._space_id = space.get("spaceId") or space.get("id")
                    logger.info(f"Reusing existing  space after 409: {self._config.space_name} ({self._space_id})")
                    return self._space_id  # type: ignore[return-value]
            # If still not found after 409, raise the original error
            response.raise_for_status()

        response.raise_for_status()
        result = response.json()
        self._space_id = result.get("spaceId") or result.get("id")
        logger.info(f"Created  space: {self._config.space_name} ({self._space_id})")
        return self._space_id  # type: ignore[return-value]

    # ── Core Memory Interface ──────────────────────────────────────────

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        """Update the model context with relevant memories from .

        Extracts the query from the last message in the context, performs a
        semantic retrieval against , and injects matching results as
        a system message.
        """
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message)

        query_results = await self.query(query_text)

        if query_results.results:
            memory_strings = [f"{i}. {str(memory.content)}" for i, memory in enumerate(query_results.results, 1)]
            memory_context = "\nRelevant memory content:\n" + "\n".join(memory_strings)
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=query_results)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add a text memory to .

        For file-based memories (PDF, images, etc.), use :meth:`add_file` instead.

        Args:
            content: The memory content to add. Only text content is supported via this method.
            cancellation_token: Optional token to cancel the operation.
        """
        space_id = await self._ensure_space()
        client = self._ensure_client()

        text = content.content
        if not isinstance(text, str):
            raise ValueError(
                "Memory.add() only supports text content. "
                "Use add_file() for binary files (PDF, images, etc.)."
            )

        request_body: Dict[str, Any] = {
            "spaceId": space_id,
            "contentType": "text/plain",
            "originalContent": text,
        }

        # Merge metadata
        merged_metadata: Dict[str, Any] = {}
        if self._config.metadata:
            merged_metadata.update(self._config.metadata)
        if content.metadata:
            merged_metadata.update(content.metadata)
        if merged_metadata:
            request_body["metadata"] = merged_metadata

        response = await client.post(
            f"{self._base_url}/v1/memories",
            headers=self._headers,
            json=request_body,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Created  memory: {result.get('memoryId')}")

    async def add_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        cancellation_token: CancellationToken | None = None,
    ) -> Dict[str, Any]:
        """Add a file (PDF, DOCX, image, etc.) as a memory to .

        Mirrors the reference integration's file handling: auto-detects content type
        from the file extension, encodes binary files as base64, and decodes text
        files to UTF-8.

        Args:
            file_path: Path to the file to store as memory.
            metadata: Optional metadata to attach to the memory.
            cancellation_token: Optional token to cancel the operation.

        Returns:
            Dict containing memoryId, spaceId, status, contentType, and fileName.
        """
        space_id = await self._ensure_space()
        client = self._ensure_client()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect MIME type from extension (mirrors reference integration)
        mime_type = _get_mime_type(path.suffix) or "application/octet-stream"

        request_body: Dict[str, Any] = {
            "spaceId": space_id,
            "contentType": mime_type,
        }

        file_bytes = path.read_bytes()

        if mime_type.startswith("text/"):
            # Text files: decode and send as originalContent
            request_body["originalContent"] = file_bytes.decode("utf-8")
        else:
            # Binary files (PDF, images, etc.): send as base64
            request_body["originalContentB64"] = base64.b64encode(file_bytes).decode("ascii")

        # Merge metadata
        merged_metadata: Dict[str, Any] = {}
        if self._config.metadata:
            merged_metadata.update(self._config.metadata)
        if metadata:
            merged_metadata.update(metadata)
        if merged_metadata:
            request_body["metadata"] = merged_metadata

        response = await client.post(
            f"{self._base_url}/v1/memories",
            headers=self._headers,
            json=request_body,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Created  file memory: {result.get('memoryId')} ({path.name})")

        return {
            "memoryId": result.get("memoryId"),
            "spaceId": result.get("spaceId"),
            "status": result.get("processingStatus", "PENDING"),
            "contentType": mime_type,
            "fileName": path.name,
        }

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Query  for semantically similar memories.

        Mirrors the reference integration's retrieve-memories logic including
        wait-for-indexing polling, NDJSON/SSE response parsing, and optional
        post-processor configuration.

        Args:
            query: The search query (text string or MemoryContent).
            cancellation_token: Optional token to cancel the operation.
            **kwargs: Additional parameters (space_ids to override default space).

        Returns:
            MemoryQueryResult containing matching memory chunks.
        """
        space_id = await self._ensure_space()
        client = self._ensure_client()

        query_text = query if isinstance(query, str) else str(query.content)

        # Build space keys — allow override via kwargs
        space_ids: List[str] = kwargs.get("space_ids", [space_id])
        space_keys = [{"spaceId": sid} for sid in space_ids]

        request_body: Dict[str, Any] = {
            "message": query_text,
            "spaceKeys": space_keys,
            "requestedSize": self._config.max_results,
            "fetchMemory": self._config.include_memory_definition,
        }

        # Add post-processor config if configured (mirrors reference)
        pp = self._config.post_processor
        if pp and (pp.reranker_id or pp.llm_id):
            config: Dict[str, Any] = {}
            if pp.reranker_id:
                config["reranker_id"] = pp.reranker_id
            if pp.llm_id:
                config["llm_id"] = pp.llm_id
            if pp.relevance_threshold is not None:
                config["relevance_threshold"] = pp.relevance_threshold
            if pp.llm_temperature is not None:
                config["llm_temp"] = pp.llm_temperature
            config["max_results"] = self._config.max_results
            if pp.chronological_resort:
                config["chronological_resort"] = True

            request_body["postProcessor"] = {
                "name": "com..retrieval.postprocess.ChatPostProcessorFactory",
                "config": config,
            }

        # Retrieval with wait-for-indexing polling (mirrors reference)
        max_wait_seconds = 60
        poll_interval_seconds = 5
        should_wait = self._config.wait_for_indexing
        start_time = asyncio.get_event_loop().time()
        last_results: List[MemoryContent] = []

        retrieve_headers = {
            **self._headers,
            "Accept": "application/x-ndjson",
        }

        while True:
            response = await client.post(
                f"{self._base_url}/v1/memories:retrieve",
                headers=retrieve_headers,
                json=request_body,
            )
            response.raise_for_status()

            results, memories, abstract_reply = _parse_retrieve_response(response.text)

            last_results = []
            for item in results:
                chunk_text = item.get("chunkText", "")
                meta: Dict[str, Any] = {
                    "chunkId": item.get("chunkId"),
                    "memoryId": item.get("memoryId"),
                    "relevanceScore": item.get("relevanceScore"),
                    "memoryIndex": item.get("memoryIndex"),
                }
                if abstract_reply:
                    meta["abstractReply"] = abstract_reply
                last_results.append(
                    MemoryContent(
                        content=chunk_text,
                        mime_type=MemoryMimeType.TEXT,
                        metadata=meta,
                    )
                )

            if last_results or not should_wait:
                break

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= max_wait_seconds:
                logger.warning("No results found after waiting 60 seconds for indexing")
                break

            await asyncio.sleep(poll_interval_seconds)

        return MemoryQueryResult(results=last_results)

    async def clear(self) -> None:
        """Clear is not directly supported by  API.

        To remove all memories, delete them individually or delete the space.
        """
        logger.warning(
            " does not support bulk clear. "
            "Use delete_memory() for individual memories or recreate the space."
        )

    async def close(self) -> None:
        """Clean up HTTP client resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        self._space_id = None

    # ── Extended  Operations ────────────────────────────────────

    async def get_memory(
        self,
        memory_id: str,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """Fetch a specific memory by ID.

        Mirrors the reference integration's get-memory action.

        Args:
            memory_id: The UUID of the memory to fetch.
            include_content: Whether to also fetch the original document content.

        Returns:
            Dict containing memory metadata and optionally its content.
        """
        client = self._ensure_client()

        response = await client.get(
            f"{self._base_url}/v1/memories/{memory_id}",
            headers=self._headers,
        )
        response.raise_for_status()
        result: Dict[str, Any] = {"memory": response.json()}

        if include_content:
            try:
                content_response = await client.get(
                    f"{self._base_url}/v1/memories/{memory_id}/content",
                    headers=self._headers,
                )
                content_response.raise_for_status()
                result["content"] = content_response.json()
            except Exception as e:
                result["contentError"] = f"Failed to fetch content: {e}"

        return result

    async def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Permanently delete a memory and its associated chunks and embeddings.

        Mirrors the reference integration's delete-memory action.

        Args:
            memory_id: The UUID of the memory to delete.

        Returns:
            Dict with memoryId and success status.
        """
        client = self._ensure_client()

        response = await client.delete(
            f"{self._base_url}/v1/memories/{memory_id}",
            headers=self._headers,
        )
        response.raise_for_status()
        logger.info(f"Deleted  memory: {memory_id}")
        return {"memoryId": memory_id, "success": True}

    async def list_embedders(self) -> List[Dict[str, Any]]:
        """List available embedder models from the  API.

        This is the equivalent of the reference integration's embedder dropdown,
        exposed as a programmatic helper for AutoGen users to discover available
        embedder IDs before configuring a space.

        Returns:
            List of embedder dicts with embedderId, displayName, modelIdentifier.
        """
        client = self._ensure_client()

        response = await client.get(
            f"{self._base_url}/v1/embedders",
            headers=self._headers,
        )
        response.raise_for_status()
        body = response.json()
        return body if isinstance(body, list) else body.get("embedders", [])

    async def list_spaces(self) -> List[Dict[str, Any]]:
        """List all available spaces from the  API.

        This is the equivalent of the reference integration's space dropdown,
        exposed as a programmatic helper for AutoGen users.

        Returns:
            List of space dicts with spaceId, name, etc.
        """
        client = self._ensure_client()

        response = await client.get(
            f"{self._base_url}/v1/spaces",
            headers=self._headers,
        )
        response.raise_for_status()
        body = response.json()
        return body if isinstance(body, list) else body.get("spaces", [])

    async def create_space(
        self,
        name: Optional[str] = None,
        embedder_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Explicitly create or reuse a space.

        Mirrors the reference integration's create-space action. Normally called
        automatically when adding memories, but available for explicit control.

        Args:
            name: Space name (defaults to config space_name).
            embedder_id: Embedder ID (defaults to config embedder_id).

        Returns:
            Dict with spaceId, name, reused flag, and chunking config.
        """
        space_name = name or self._config.space_name
        emb_id = embedder_id or self._config.embedder_id
        client = self._ensure_client()

        # Check for existing space
        try:
            response = await client.get(
                f"{self._base_url}/v1/spaces",
                headers=self._headers,
            )
            response.raise_for_status()
            body = response.json()
            spaces = body if isinstance(body, list) else body.get("spaces", [])
            for space in spaces:
                if space.get("name") == space_name:
                    sid = space.get("spaceId") or space.get("id")
                    # Cache if this is the default space
                    if space_name == self._config.space_name:
                        self._space_id = sid
                    return {
                        "spaceId": sid,
                        "name": space_name,
                        "embedderId": emb_id,
                        "reused": True,
                    }
        except Exception:
            pass

        # Create new space
        chunking = self._config.chunking
        request_body: Dict[str, Any] = {
            "name": space_name,
            "spaceEmbedders": [{"embedderId": emb_id, "defaultRetrievalWeight": 1.0}],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunking.chunk_size,
                    "chunkOverlap": chunking.chunk_overlap,
                    "separators": chunking.separators,
                    "keepStrategy": chunking.keep_strategy,
                    "separatorIsRegex": chunking.separator_is_regex,
                    "lengthMeasurement": chunking.length_measurement,
                },
            },
        }

        response = await client.post(
            f"{self._base_url}/v1/spaces",
            headers=self._headers,
            json=request_body,
        )
        response.raise_for_status()
        result = response.json()
        sid = result.get("spaceId") or result.get("id")

        # Cache if this is the default space
        if space_name == self._config.space_name:
            self._space_id = sid

        return {
            "spaceId": sid,
            "name": result.get("name"),
            "embedderId": emb_id,
            "chunkingConfig": request_body["defaultChunkingConfig"],
            "reused": False,
        }

    # ── Serialization ──────────────────────────────────────────────────

    def _to_config(self) -> GoodMemMemoryConfig:
        return self._config

    @classmethod
    def _from_config(cls, config: GoodMemMemoryConfig) -> Self:
        return cls(config=config)


# ── Helper Functions ──────────────────────────────────────────────────


def _get_mime_type(extension: str) -> Optional[str]:
    """Auto-detect MIME type from file extension.

    Mirrors the reference integration's getMimeType function.
    """
    mime_types: Dict[str, str] = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".txt": "text/plain",
        ".html": "text/html",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".json": "application/json",
        ".xml": "application/xml",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".ppt": "application/vnd.ms-powerpoint",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    return mime_types.get(extension.lower())


def _parse_retrieve_response(response_text: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Any]:
    """Parse  NDJSON/SSE retrieval response.

    Mirrors the reference integration's response parsing logic.

    Returns:
        Tuple of (results, memories, abstract_reply).
    """
    results: List[Dict[str, Any]] = []
    memories: List[Dict[str, Any]] = []
    abstract_reply: Any = None

    lines = response_text.strip().split("\n")
    for line in lines:
        json_str = line.strip()
        if not json_str:
            continue

        # Handle SSE format: extract JSON from "data: {...}" lines
        if json_str.startswith("data:"):
            json_str = json_str[5:].strip()
        # Skip SSE event type lines
        if json_str.startswith("event:") or not json_str:
            continue

        try:
            item = json.loads(json_str)

            if item.get("memoryDefinition"):
                memories.append(item["memoryDefinition"])
            elif item.get("abstractReply"):
                abstract_reply = item["abstractReply"]
            elif item.get("retrievedItem"):
                chunk_data = item["retrievedItem"].get("chunk", {})
                chunk = chunk_data.get("chunk", {})
                results.append({
                    "chunkId": chunk.get("chunkId"),
                    "chunkText": chunk.get("chunkText"),
                    "memoryId": chunk.get("memoryId"),
                    "relevanceScore": chunk_data.get("relevanceScore"),
                    "memoryIndex": chunk_data.get("memoryIndex"),
                })
        except json.JSONDecodeError:
            # Skip non-JSON lines (SSE event types, close events)
            continue

    return results, memories, abstract_reply
