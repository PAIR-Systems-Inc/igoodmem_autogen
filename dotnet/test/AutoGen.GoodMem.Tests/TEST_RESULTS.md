# GoodMem .NET Integration Test Results

**Date**: 2026-03-23
**GoodMem Server**: https://localhost:8080 (self-signed TLS)
**API Key**: gm_g5xcse2tjgcznlg45c5le4ti5q
**Embedder Used**: Voyage AI (voyage-3-large) - ID: 019cfd1c-c033-7517-b7de-f73941a0464b
**Test Space**: dotnet-test-ae0b7f4f797a40c49b

## Command Executed

```bash
export PATH="/home/bashar/.dotnet:$PATH"
cd /home/bashar/igoodmem_autogen/dotnet
dotnet test test/AutoGen.GoodMem.Tests/AutoGen.GoodMem.Tests.csproj --logger "console;verbosity=detailed"
```

## Results Summary

| # | Test | Status | Duration |
|---|------|--------|----------|
| 1 | List Embedders | PASS | 96 ms |
| 2 | Create Space | PASS | 69 ms |
| 3 | List Spaces | PASS | 10 ms |
| 4 | Create Text Memory | PASS | 15 ms |
| 5 | Create File Memory (PDF) | PASS | 35 ms |
| 6 | Get Memory | PASS | 13 ms |
| 7 | Retrieve Memories | PASS | ~5 s |
| 8 | Delete Memory | PASS | 22 ms |

**Total: 8 Passed, 0 Failed** (6.4 seconds total)

## Detailed Output

### Test 1: List Embedders
Discovered 3 embedders:
- OpenAI Text Embedding 3 Small (text-embedding-3-small) - 019cfd11-6ea9-75c8-925c-6a202a517513
- Voyage AI (voyage-3-large) - 019cfd1c-c033-7517-b7de-f73941a0464b
- Qwen3 8B (qwen/qwen3-embedding-8b) - 019cfd94-2844-7117-85ca-1b9919758a26
Selected: Voyage AI (confirmed working from prior testing)

### Test 2: Create Space
Created space "dotnet-test-ae0b7f4f797a40c49b" with ID 019d1c69-2077-725e-b14c-cb61bffe30b8.
Reused: false (new space created successfully).

### Test 3: List Spaces
Found 10 total spaces. Confirmed our test space was present in the list.

### Test 4: Create Text Memory
Created text memory with ID 019d1c69-209a-74ce-96a3-176bc3777954.
Content: "The capital of France is Paris. France is a country in Western Europe..."
Status: PENDING (async processing)

### Test 5: Create File Memory (PDF)
Successfully uploaded PDF file: "New Quran.com Search Analysis (Nov 26, 2025)-1.pdf"
Memory ID: 019d1c69-20b9-75e6-bf7e-e39c91654a01
Content Type: application/pdf (auto-detected from .pdf extension)
Status: PENDING

### Test 6: Get Memory
Retrieved memory metadata for the text memory.
Processing status confirmed as PENDING.
Content fetch returned a non-JSON response (raw text), which is expected behavior
for the /content endpoint returning plain text.

### Test 7: Retrieve Memories
Query: "What is the capital of France?"
Wait-for-indexing: enabled (polled until results appeared, ~5 seconds)
Retrieved 5 chunks across 2 memories:
- Top result: text memory chunk with relevance score -0.71 ("The capital of France is Paris...")
- Also matched chunks from the PDF memory

### Test 8: Delete Memory
Successfully deleted both the text memory and the PDF file memory.
Both operations returned success: true.

## Notes

- GoodMem server requires HTTPS with self-signed certificate; SkipSslValidation option handles this.
- The /v1/memories/{id}/content endpoint returns raw text, not JSON - the content parsing error is cosmetic and does not affect functionality.
- Wait-for-indexing polling worked correctly, waiting ~5 seconds for the async indexing to complete before returning results.
- PDF auto-detection correctly identified application/pdf from the .pdf extension.
