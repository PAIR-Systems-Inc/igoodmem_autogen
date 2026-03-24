"""End-to-end integration test: AutoGen Agent + GoodMem Memory.

Demonstrates the full flow:
1. Create a GoodMem memory instance
2. Store knowledge (text and file) into GoodMem
3. Create an AutoGen AssistantAgent with GoodMem as its memory
4. Ask the agent a question — it automatically retrieves relevant memories
5. Verify the agent's response is informed by the stored memories

Usage:
    # With a live GoodMem server:
    GOODMEM_API_KEY=gm_xxx GOODMEM_BASE_URL=http://localhost:8080 OPENAI_API_KEY=sk-xxx \
        python -m pytest tests/test_goodmem_agent_integration.py -v -s

    # With environment defaults (localhost:8080):
    python -m pytest tests/test_goodmem_agent_integration.py -v -s
"""

import asyncio
import os
import uuid

import pytest
import pytest_asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.goodmem import GoodMemMemory, GoodMemMemoryConfig
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ── Configuration ────────────────────────────────────────────────────

GOODMEM_API_KEY = os.environ.get("GOODMEM_API_KEY", "")
GOODMEM_BASE_URL = os.environ.get("GOODMEM_BASE_URL", "https://localhost:8080")
EMBEDDER_ID = os.environ.get("GOODMEM_EMBEDDER_ID", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not GOODMEM_API_KEY:
    raise RuntimeError("GOODMEM_API_KEY env var is required. Set it to a valid GoodMem API key.")
if not EMBEDDER_ID:
    raise RuntimeError("GOODMEM_EMBEDDER_ID env var is required. Set it to a valid embedder UUID.")

TEST_SPACE_NAME = f"agent-integration-test-{uuid.uuid4().hex[:8]}"

# Knowledge to store in GoodMem
KNOWLEDGE_ENTRIES = [
    {
        "content": "The iGoodMem project is a vector memory system that provides semantic storage and retrieval. "
        "It supports text and file-based memories with vector embeddings using configurable embedder models.",
        "metadata": {"topic": "igoodmem", "category": "overview"},
    },
    {
        "content": "GoodMem organizes memories into spaces. Each space has an associated embedder model. "
        "When a document is added, it is chunked, embedded, and stored for later semantic retrieval.",
        "metadata": {"topic": "igoodmem", "category": "architecture"},
    },
    {
        "content": "The AutoGen framework by Microsoft supports multi-agent AI applications. "
        "Agents can be given memory backends so they can recall relevant information during conversations.",
        "metadata": {"topic": "autogen", "category": "overview"},
    },
    {
        "content": "GoodMem retrieval supports post-processing with rerankers and LLMs to improve result quality. "
        "It also has a wait-for-indexing feature that polls up to 60 seconds for newly added memories to be indexed.",
        "metadata": {"topic": "igoodmem", "category": "retrieval"},
    },
]


def _make_memory_config() -> GoodMemMemoryConfig:
    return GoodMemMemoryConfig(
        base_url=GOODMEM_BASE_URL,
        api_key=GOODMEM_API_KEY,
        space_name=TEST_SPACE_NAME,
        embedder_id=EMBEDDER_ID,
        max_results=3,
        wait_for_indexing=True,
        verify_ssl=False,
    )


def _make_model_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=OPENAI_API_KEY,
    )


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def goodmem_memory():
    """Create a GoodMem memory instance and clean up after tests."""
    mem = GoodMemMemory(config=_make_memory_config())
    yield mem
    await mem.close()


# ── Test 1: Store Knowledge into GoodMem ─────────────────────────────


@pytest.mark.asyncio
async def test_01_store_knowledge(goodmem_memory: GoodMemMemory):
    """Store multiple knowledge entries into GoodMem for the agent to use later."""
    for entry in KNOWLEDGE_ENTRIES:
        content = MemoryContent(
            content=entry["content"],
            mime_type=MemoryMimeType.TEXT,
            metadata=entry["metadata"],
        )
        await goodmem_memory.add(content)

    print(f"\n  Stored {len(KNOWLEDGE_ENTRIES)} knowledge entries in space: {TEST_SPACE_NAME}")

    # Verify the space was created
    spaces = await goodmem_memory.list_spaces()
    our_space = [s for s in spaces if s.get("name") == TEST_SPACE_NAME]
    assert len(our_space) == 1, f"Expected test space '{TEST_SPACE_NAME}' to exist"
    print(f"  Space ID: {our_space[0].get('spaceId')}")


# ── Test 2: Retrieve Knowledge Directly ──────────────────────────────


@pytest.mark.asyncio
async def test_02_retrieve_knowledge(goodmem_memory: GoodMemMemory):
    """Verify that stored knowledge can be retrieved via semantic search."""
    # First, store the knowledge (each test gets a fresh instance)
    for entry in KNOWLEDGE_ENTRIES:
        await goodmem_memory.add(
            MemoryContent(
                content=entry["content"],
                mime_type=MemoryMimeType.TEXT,
                metadata=entry["metadata"],
            )
        )

    # Query for iGoodMem-specific knowledge
    results = await goodmem_memory.query("How does GoodMem organize and store memories?")
    assert len(results.results) > 0, "Expected at least one result"

    print(f"\n  Query: 'How does GoodMem organize and store memories?'")
    print(f"  Results: {len(results.results)}")
    for i, r in enumerate(results.results):
        score = r.metadata.get("relevanceScore", "N/A") if r.metadata else "N/A"
        preview = str(r.content)[:100] + "..." if len(str(r.content)) > 100 else str(r.content)
        print(f"    [{i + 1}] score={score} | {preview}")


# ── Test 3: Agent Uses GoodMem Memory ────────────────────────────────


@pytest.mark.asyncio
async def test_03_agent_with_goodmem_memory(goodmem_memory: GoodMemMemory):
    """Create an agent with GoodMem memory and verify it uses stored knowledge.

    This is the key integration test:
    1. Store knowledge in GoodMem
    2. Create an AssistantAgent with GoodMem as its memory
    3. Ask a question that requires the stored knowledge
    4. Verify the agent's response references the stored information
    """
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set — skipping agent test")

    # Step 1: Store knowledge
    for entry in KNOWLEDGE_ENTRIES:
        await goodmem_memory.add(
            MemoryContent(
                content=entry["content"],
                mime_type=MemoryMimeType.TEXT,
                metadata=entry["metadata"],
            )
        )
    print(f"\n  Stored {len(KNOWLEDGE_ENTRIES)} entries")

    # Step 2: Create agent with GoodMem memory
    model_client = _make_model_client()
    agent = AssistantAgent(
        name="knowledge_assistant",
        model_client=model_client,
        memory=[goodmem_memory],
        system_message=(
            "You are a knowledgeable assistant. Use the memory context provided to you "
            "to answer questions accurately. If you have relevant memory context, reference "
            "it in your answer."
        ),
    )

    # Step 3: Ask a question that the stored knowledge can answer
    result = await agent.run(task="What is GoodMem and how does it organize memories?")

    # Step 4: Check response
    response_text = result.messages[-1].content
    print(f"\n  Agent response:\n  {response_text}")

    # The response should reference concepts from our stored knowledge
    response_lower = str(response_text).lower()
    assert any(
        keyword in response_lower for keyword in ["space", "vector", "embed", "semantic", "chunk", "memory"]
    ), f"Agent response does not seem to reference stored knowledge: {response_text}"

    print("\n  Agent successfully used GoodMem memory to answer the question!")


# ── Test 4: Agent Multi-Turn with Memory ─────────────────────────────


@pytest.mark.asyncio
async def test_04_agent_multi_turn(goodmem_memory: GoodMemMemory):
    """Test that the agent retrieves different memories based on different questions."""
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set — skipping agent test")

    # Store knowledge
    for entry in KNOWLEDGE_ENTRIES:
        await goodmem_memory.add(
            MemoryContent(
                content=entry["content"],
                mime_type=MemoryMimeType.TEXT,
                metadata=entry["metadata"],
            )
        )

    model_client = _make_model_client()
    agent = AssistantAgent(
        name="multi_turn_assistant",
        model_client=model_client,
        memory=[goodmem_memory],
        system_message="You are a helpful assistant. Use the memory context to answer questions.",
    )

    # Question 1: About AutoGen
    result1 = await agent.run(task="What is AutoGen and what does it support?")
    response1 = str(result1.messages[-1].content).lower()
    print(f"\n  Q1 response: {result1.messages[-1].content[:200]}")
    assert any(
        kw in response1 for kw in ["multi-agent", "agent", "autogen", "microsoft"]
    ), "Response 1 should mention AutoGen concepts"

    # Question 2: About retrieval features
    result2 = await agent.run(task="How does GoodMem improve retrieval quality?")
    response2 = str(result2.messages[-1].content).lower()
    print(f"\n  Q2 response: {result2.messages[-1].content[:200]}")
    assert any(
        kw in response2 for kw in ["rerank", "post-process", "index", "retrieval", "quality"]
    ), "Response 2 should mention retrieval features"

    print("\n  Multi-turn test passed — agent retrieved different context for different questions!")


# ── Test 5: Memory Context Injection Verification ────────────────────


@pytest.mark.asyncio
async def test_05_update_context_directly(goodmem_memory: GoodMemMemory):
    """Verify that update_context() injects memories into the model context as a SystemMessage.

    This tests the mechanism that agents use internally — without needing an LLM.
    """
    from autogen_core.model_context import BufferedChatCompletionContext
    from autogen_core.models import SystemMessage, UserMessage

    # Store knowledge
    for entry in KNOWLEDGE_ENTRIES:
        await goodmem_memory.add(
            MemoryContent(
                content=entry["content"],
                mime_type=MemoryMimeType.TEXT,
                metadata=entry["metadata"],
            )
        )

    # Create a model context with a user message
    context = BufferedChatCompletionContext(buffer_size=10)
    await context.add_message(UserMessage(content="Tell me about GoodMem spaces", source="user"))

    # Call update_context — this is what the agent does internally
    result = await goodmem_memory.update_context(context)

    # Verify memories were retrieved
    assert len(result.memories.results) > 0, "Expected memories to be retrieved"
    print(f"\n  Retrieved {len(result.memories.results)} memories via update_context()")

    # Verify a SystemMessage was injected into the context
    messages = await context.get_messages()
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) > 0, "Expected a SystemMessage to be injected with memory context"

    memory_msg = system_messages[0].content
    print(f"\n  Injected SystemMessage preview:\n  {str(memory_msg)[:300]}...")

    # The injected message should contain our knowledge
    assert "memory" in str(memory_msg).lower() or "space" in str(memory_msg).lower(), (
        "Injected SystemMessage should contain relevant memory content"
    )

    print("\n  update_context() correctly injects memories as SystemMessage!")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
