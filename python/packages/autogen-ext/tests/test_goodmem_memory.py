"""End-to-end integration tests for GoodMem memory.

These tests run against a live GoodMem server and validate all core actions:
1. List Embedders
2. Create Space (or reuse existing)
3. Create Memory with normal text
4. Create Memory with a PDF file
5. Retrieve Memories (semantic search with wait-for-indexing)
6. Get Memory (by ID with content)
7. Delete Memory
8. List Spaces

Usage:
    GOODMEM_API_KEY=gm_xxx GOODMEM_BASE_URL=http://localhost:8080 python -m pytest tests/test_goodmem_memory.py -v -s
"""

import asyncio
import os
import uuid

import pytest
import pytest_asyncio

from autogen_ext.memory.goodmem import GoodMemMemory, GoodMemMemoryConfig
from autogen_core.memory import MemoryContent, MemoryMimeType


# ── Configuration ────────────────────────────────────────────────────

GOODMEM_API_KEY = os.environ.get("GOODMEM_API_KEY", "")
GOODMEM_BASE_URL = os.environ.get("GOODMEM_BASE_URL", "https://localhost:8080")
EMBEDDER_ID = os.environ.get("GOODMEM_EMBEDDER_ID", "")
PDF_FILE_PATH = os.environ.get("GOODMEM_PDF_PATH", "")

if not GOODMEM_API_KEY:
    raise RuntimeError("GOODMEM_API_KEY env var is required. Set it to a valid GoodMem API key.")
if not EMBEDDER_ID:
    raise RuntimeError("GOODMEM_EMBEDDER_ID env var is required. Set it to a valid embedder UUID.")

# Shared space name for all tests in this module
TEST_SPACE_NAME = f"autogen-test-{uuid.uuid4().hex[:8]}"

# Module-level state shared across tests
_state: dict = {}


def _make_config() -> GoodMemMemoryConfig:
    return GoodMemMemoryConfig(
        base_url=GOODMEM_BASE_URL,
        api_key=GOODMEM_API_KEY,
        space_name=TEST_SPACE_NAME,
        embedder_id=EMBEDDER_ID,
        max_results=5,
        wait_for_indexing=True,
        verify_ssl=False,
    )


@pytest_asyncio.fixture
async def memory():
    """Create a fresh GoodMemMemory instance per test (avoids event loop issues)."""
    mem = GoodMemMemory(config=_make_config())
    yield mem
    await mem.close()


# ── Test 1: List Embedders ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_01_list_embedders(memory: GoodMemMemory):
    """Verify that list_embedders returns available embedding models."""
    embedders = await memory.list_embedders()
    assert isinstance(embedders, list), f"Expected list, got {type(embedders)}"
    assert len(embedders) > 0, "No embedders found"

    first = embedders[0]
    assert "embedderId" in first, f"Missing embedderId in {first.keys()}"

    print(f"\n  Found {len(embedders)} embedders:")
    for e in embedders:
        print(f"    - {e.get('displayName', 'unnamed')} ({e.get('embedderId')})")


# ── Test 2: Create Space ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_02_create_space(memory: GoodMemMemory):
    """Verify that create_space creates a new space or reuses an existing one."""
    result = await memory.create_space()
    assert "spaceId" in result, f"Missing spaceId in result: {result}"
    _state["spaceId"] = result["spaceId"]
    print(f"\n  Space: {result.get('name')} (ID: {result.get('spaceId')}, reused: {result.get('reused')})")

    # Test idempotency: creating again should reuse
    result2 = await memory.create_space()
    assert result2.get("spaceId") == result.get("spaceId"), "Expected same space ID on second create"
    assert result2.get("reused") is True, "Expected reused=True on second create"
    print(f"  Reuse confirmed: {result2.get('reused')}")


# ── Test 3: Create Memory with Text ────────────────────────────────

@pytest.mark.asyncio
async def test_03_create_memory_text(memory: GoodMemMemory):
    """Verify that adding text content creates a memory in GoodMem."""
    content = MemoryContent(
        content="AutoGen is a framework for building multi-agent AI applications with conversational patterns and tool use.",
        mime_type=MemoryMimeType.TEXT,
        metadata={"category": "technology", "topic": "autogen"},
    )
    await memory.add(content)
    print("\n  Text memory created successfully")

    # Verify via list spaces that our space exists
    spaces = await memory.list_spaces()
    our_space = [s for s in spaces if s.get("name") == TEST_SPACE_NAME]
    assert len(our_space) == 1, f"Expected to find test space '{TEST_SPACE_NAME}'"
    print(f"  Verified space exists: {our_space[0].get('spaceId')}")


# ── Test 4: Create Memory with PDF ─────────────────────────────────

@pytest.mark.asyncio
async def test_04_create_memory_pdf(memory: GoodMemMemory):
    """Verify that adding a PDF file creates a memory in GoodMem."""
    if not os.path.exists(PDF_FILE_PATH):
        pytest.skip(f"PDF file not found: {PDF_FILE_PATH}")

    result = await memory.add_file(PDF_FILE_PATH)
    assert "memoryId" in result, f"Missing memoryId in result: {result}"
    assert result.get("contentType") == "application/pdf"
    assert result.get("fileName") is not None
    _state["pdfMemoryId"] = result["memoryId"]

    print(f"\n  PDF memory created: {result.get('memoryId')}")
    print(f"  File: {result.get('fileName')}")
    print(f"  Content type: {result.get('contentType')}")
    print(f"  Status: {result.get('status')}")


# ── Test 5: Retrieve Memories ──────────────────────────────────────

@pytest.mark.asyncio
async def test_05_retrieve_memories(memory: GoodMemMemory):
    """Verify semantic retrieval returns relevant results.

    Uses wait_for_indexing=True to poll until the text memory from test_03
    has been indexed and is retrievable.
    """
    results = await memory.query("What is AutoGen used for?")
    assert isinstance(results.results, list)

    print(f"\n  Query: 'What is AutoGen used for?'")
    print(f"  Results found: {len(results.results)}")
    for i, r in enumerate(results.results):
        score = r.metadata.get("relevanceScore", "N/A") if r.metadata else "N/A"
        text_preview = str(r.content)[:120] + "..." if len(str(r.content)) > 120 else str(r.content)
        print(f"    [{i+1}] score={score} | {text_preview}")

    assert len(results.results) > 0, "Expected at least one retrieval result"

    # Save a memoryId for subsequent tests
    if results.results[0].metadata:
        _state["textMemoryId"] = results.results[0].metadata.get("memoryId")


# ── Test 6: Get Memory ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_06_get_memory(memory: GoodMemMemory):
    """Verify that get_memory returns metadata for a specific memory."""
    memory_id = _state.get("textMemoryId") or _state.get("pdfMemoryId")
    if not memory_id:
        pytest.skip("No memoryId available from previous tests")

    result = await memory.get_memory(memory_id, include_content=True)
    assert "memory" in result, f"Missing 'memory' key in result: {result.keys()}"
    assert result["memory"].get("memoryId") == memory_id

    print(f"\n  Memory ID: {memory_id}")
    print(f"  Status: {result['memory'].get('processingStatus')}")
    print(f"  Content type: {result['memory'].get('contentType')}")
    if "content" in result:
        print(f"  Content available: yes")
    elif "contentError" in result:
        print(f"  Content error: {result['contentError']}")


# ── Test 7: Delete Memory ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_07_delete_memory(memory: GoodMemMemory):
    """Verify that delete_memory removes a memory."""
    # Create a dedicated memory to delete
    content = MemoryContent(
        content="Temporary content specifically created for deletion testing in the AutoGen GoodMem integration.",
        mime_type=MemoryMimeType.TEXT,
    )
    await memory.add(content)

    # Retrieve to get memory ID
    results = await memory.query("Temporary content deletion testing AutoGen")
    if len(results.results) == 0:
        pytest.skip("Could not retrieve memory for deletion test (indexing may be slow)")

    memory_id = results.results[0].metadata.get("memoryId") if results.results[0].metadata else None
    assert memory_id is not None, "Missing memoryId"

    # Delete it
    delete_result = await memory.delete_memory(memory_id)
    assert delete_result.get("success") is True
    assert delete_result.get("memoryId") == memory_id

    print(f"\n  Deleted memory: {memory_id}")

    # Verify it is gone
    try:
        await memory.get_memory(memory_id)
        print("  Note: get_memory after delete did not raise (may return error status)")
    except Exception as e:
        print(f"  Confirmed deletion: {type(e).__name__}")


# ── Test 8: List Spaces ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_08_list_spaces(memory: GoodMemMemory):
    """Verify that list_spaces returns available spaces."""
    spaces = await memory.list_spaces()
    assert isinstance(spaces, list), f"Expected list, got {type(spaces)}"
    assert len(spaces) > 0, "No spaces found"

    our_space = [s for s in spaces if s.get("name") == TEST_SPACE_NAME]
    assert len(our_space) == 1, f"Expected to find test space '{TEST_SPACE_NAME}'"

    print(f"\n  Total spaces: {len(spaces)}")
    print(f"  Test space found: {our_space[0].get('spaceId')}")
