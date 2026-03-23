// Copyright (c) Microsoft Corporation. All rights reserved.
// GoodMemIntegrationTests.cs

using FluentAssertions;
using Microsoft.AutoGen.Extensions.GoodMem;
using Microsoft.Extensions.Logging;
using Xunit;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace AutoGen.GoodMem.Tests;

/// <summary>
/// Integration tests for GoodMem client against a live GoodMem server.
/// These tests hit the real GoodMem API at http://localhost:8080.
/// They require a running GoodMem server and a valid API key.
///
/// Test execution order is controlled to ensure proper setup/teardown:
/// 1. List Embedders (discover available embedders)
/// 2. Create Space
/// 3. List Spaces (verify space was created)
/// 4. Create Text Memory
/// 5. Create File Memory (PDF)
/// 6. Get Memory
/// 7. Retrieve Memories (with wait-for-indexing)
/// 8. Delete Memory
/// </summary>
[TestCaseOrderer("AutoGen.GoodMem.Tests.PriorityOrderer", "AutoGen.GoodMem.Tests")]
[Trait("Category", "Integration")]
public class GoodMemIntegrationTests : IAsyncLifetime
{
    private const string BaseUrl = "https://localhost:8080";
    private const string ApiKey = "gm_g5xcse2tjgcznlg45c5le4ti5q";
    private const string PdfPath = "/home/bashar/Downloads/New Quran.com Search Analysis (Nov 26, 2025)-1.pdf";

    private readonly ITestOutputHelper _output;
    private readonly ILogger<GoodMemClient> _logger;
    private GoodMemClient _client = null!;

    // Shared state across tests
    private static string? _embedderId;
    private static string? _spaceId;
    private static string? _textMemoryId;
    private static string? _fileMemoryId;
    private static readonly string SpaceName = $"dotnet-test-{Guid.NewGuid():N}".Substring(0, 30);

    public GoodMemIntegrationTests(ITestOutputHelper output)
    {
        _output = output;
        _logger = new TestOutputLogger<GoodMemClient>(output);
    }

    private static GoodMemOptions CreateOptions(
        string? embedderId = null,
        bool waitForIndexing = true,
        int maxResults = 5)
    {
        return new GoodMemOptions
        {
            BaseUrl = BaseUrl,
            ApiKey = ApiKey,
            SpaceName = SpaceName,
            EmbedderId = embedderId ?? _embedderId ?? "placeholder",
            MaxResults = maxResults,
            WaitForIndexing = waitForIndexing,
            IncludeMemoryDefinition = true,
            SkipSslValidation = true,
        };
    }

    public Task InitializeAsync()
    {
        _client = new GoodMemClient(CreateOptions(), _logger);
        return Task.CompletedTask;
    }

    public Task DisposeAsync()
    {
        _client?.Dispose();
        return Task.CompletedTask;
    }

    [Fact, TestPriority(1)]
    public async Task Test01_ListEmbedders()
    {
        _output.WriteLine("=== Test 1: List Embedders ===");

        var embedders = await _client.ListEmbeddersAsync();

        embedders.Should().NotBeEmpty("GoodMem should have at least one embedder available");

        foreach (var e in embedders)
        {
            var id = e.TryGetValue("embedderId", out var eid) ? eid?.ToString() : "??";
            var name = e.TryGetValue("displayName", out var dn) ? dn?.ToString() :
                       e.TryGetValue("name", out var n) ? n?.ToString() : "??";
            var model = e.TryGetValue("modelIdentifier", out var mi) ? mi?.ToString() : "??";
            _output.WriteLine($"  Embedder: {name} ({model}) — ID: {id}");
        }

        // Pick the Voyage embedder if available (confirmed working), otherwise first available
        var voyageEmbedder = embedders.FirstOrDefault(e =>
        {
            var model = e.TryGetValue("modelIdentifier", out var mi) ? mi?.ToString() : "";
            return model != null && model.Contains("voyage", StringComparison.OrdinalIgnoreCase);
        });

        var selectedEmbedder = voyageEmbedder ?? embedders.First();
        _embedderId = selectedEmbedder.TryGetValue("embedderId", out var selectedId) ? selectedId?.ToString() :
                      selectedEmbedder.TryGetValue("id", out var altId) ? altId?.ToString() : null;

        _embedderId.Should().NotBeNullOrEmpty("should find an embedder ID");
        _output.WriteLine($"  Selected embedder ID: {_embedderId}");
    }

    [Fact, TestPriority(2)]
    public async Task Test02_CreateSpace()
    {
        _output.WriteLine("=== Test 2: Create Space ===");
        _embedderId.Should().NotBeNullOrEmpty("Test01 must run first to discover embedders");

        // Recreate client with correct embedder ID
        _client.Dispose();
        _client = new GoodMemClient(CreateOptions(), _logger);

        var result = await _client.CreateSpaceAsync();

        result.Should().ContainKey("spaceId");
        result["spaceId"].Should().NotBeNull();
        _spaceId = result["spaceId"]?.ToString();

        _output.WriteLine($"  Space created: {SpaceName}");
        _output.WriteLine($"  Space ID: {_spaceId}");
        _output.WriteLine($"  Reused: {result.GetValueOrDefault("reused")}");
        _output.WriteLine($"  Message: {result.GetValueOrDefault("message")}");
    }

    [Fact, TestPriority(3)]
    public async Task Test03_ListSpaces()
    {
        _output.WriteLine("=== Test 3: List Spaces ===");

        var spaces = await _client.ListSpacesAsync();

        spaces.Should().NotBeEmpty("should have at least the space we just created");

        var ourSpace = spaces.FirstOrDefault(s =>
        {
            var name = s.TryGetValue("name", out var n) ? n?.ToString() : null;
            return name == SpaceName;
        });

        ourSpace.Should().NotBeNull($"should find our space '{SpaceName}' in the list");
        _output.WriteLine($"  Found {spaces.Count} spaces total");
        _output.WriteLine($"  Our space '{SpaceName}' found: yes");
    }

    [Fact, TestPriority(4)]
    public async Task Test04_CreateTextMemory()
    {
        _output.WriteLine("=== Test 4: Create Text Memory ===");
        _embedderId.Should().NotBeNullOrEmpty("Test01 must run first");

        // Recreate client with correct embedder
        _client.Dispose();
        _client = new GoodMemClient(CreateOptions(), _logger);

        var result = await _client.CreateTextMemoryAsync(
            "The capital of France is Paris. France is a country in Western Europe known for the Eiffel Tower, fine cuisine, and its rich history. Paris has been the capital since the 10th century.");

        result.Should().ContainKey("memoryId");
        result["memoryId"].Should().NotBeNull();
        _textMemoryId = result["memoryId"]?.ToString();

        _output.WriteLine($"  Memory ID: {_textMemoryId}");
        _output.WriteLine($"  Status: {result.GetValueOrDefault("status")}");
        _output.WriteLine($"  Content Type: {result.GetValueOrDefault("contentType")}");
    }

    [Fact, TestPriority(5)]
    public async Task Test05_CreateFileMemory_PDF()
    {
        _output.WriteLine("=== Test 5: Create File Memory (PDF) ===");
        _embedderId.Should().NotBeNullOrEmpty("Test01 must run first");

        System.IO.File.Exists(PdfPath).Should().BeTrue($"PDF file should exist at {PdfPath}");

        // Recreate client with correct embedder
        _client.Dispose();
        _client = new GoodMemClient(CreateOptions(), _logger);

        var result = await _client.CreateFileMemoryAsync(PdfPath);

        result.Should().ContainKey("memoryId");
        result["memoryId"].Should().NotBeNull();
        _fileMemoryId = result["memoryId"]?.ToString();

        _output.WriteLine($"  Memory ID: {_fileMemoryId}");
        _output.WriteLine($"  Status: {result.GetValueOrDefault("status")}");
        _output.WriteLine($"  Content Type: {result.GetValueOrDefault("contentType")}");
        _output.WriteLine($"  File Name: {result.GetValueOrDefault("fileName")}");

        result["contentType"]?.ToString().Should().Be("application/pdf");
    }

    [Fact, TestPriority(6)]
    public async Task Test06_GetMemory()
    {
        _output.WriteLine("=== Test 6: Get Memory ===");
        _textMemoryId.Should().NotBeNullOrEmpty("Test04 must run first");

        var result = await _client.GetMemoryAsync(_textMemoryId!);

        result.Should().ContainKey("memory");
        var memory = result["memory"] as Dictionary<string, object?>;
        memory.Should().NotBeNull();

        _output.WriteLine($"  Memory ID: {memory?.GetValueOrDefault("memoryId")}");
        _output.WriteLine($"  Space ID: {memory?.GetValueOrDefault("spaceId")}");
        _output.WriteLine($"  Processing Status: {memory?.GetValueOrDefault("processingStatus")}");
        _output.WriteLine($"  Content Type: {memory?.GetValueOrDefault("contentType")}");

        if (result.TryGetValue("content", out var content))
        {
            _output.WriteLine($"  Content retrieved: yes (type: {content?.GetType().Name})");
        }
        else if (result.TryGetValue("contentError", out var contentError))
        {
            _output.WriteLine($"  Content error: {contentError}");
        }
    }

    [Fact, TestPriority(7)]
    public async Task Test07_RetrieveMemories()
    {
        _output.WriteLine("=== Test 7: Retrieve Memories ===");
        _embedderId.Should().NotBeNullOrEmpty("Test01 must run first");

        // Recreate client with correct embedder and wait-for-indexing enabled
        _client.Dispose();
        _client = new GoodMemClient(CreateOptions(waitForIndexing: true), _logger);

        var result = await _client.RetrieveMemoriesAsync("What is the capital of France?");

        _output.WriteLine($"  Query: What is the capital of France?");
        _output.WriteLine($"  Result Set ID: {result.ResultSetId}");
        _output.WriteLine($"  Total Results: {result.TotalResults}");
        _output.WriteLine($"  Memories: {result.Memories.Count}");

        result.Results.Should().NotBeEmpty("should find results matching 'capital of France'");

        foreach (var chunk in result.Results.Take(3))
        {
            _output.WriteLine($"  --- Chunk ---");
            _output.WriteLine($"    Chunk ID: {chunk.ChunkId}");
            _output.WriteLine($"    Memory ID: {chunk.MemoryId}");
            _output.WriteLine($"    Score: {chunk.RelevanceScore}");
            _output.WriteLine($"    Text: {chunk.ChunkText?.Substring(0, Math.Min(100, chunk.ChunkText?.Length ?? 0))}...");
        }
    }

    [Fact, TestPriority(8)]
    public async Task Test08_DeleteMemory()
    {
        _output.WriteLine("=== Test 8: Delete Memory ===");
        _textMemoryId.Should().NotBeNullOrEmpty("Test04 must run first");

        var result = await _client.DeleteMemoryAsync(_textMemoryId!);

        result.Should().ContainKey("success");
        result["success"].Should().Be(true);
        result["memoryId"]?.ToString().Should().Be(_textMemoryId);

        _output.WriteLine($"  Deleted memory: {_textMemoryId}");
        _output.WriteLine($"  Success: {result["success"]}");

        // Also delete the file memory to clean up
        if (_fileMemoryId != null)
        {
            await _client.DeleteMemoryAsync(_fileMemoryId);
            _output.WriteLine($"  Also deleted file memory: {_fileMemoryId}");
        }
    }
}

// ── Test ordering infrastructure ──────────────────────────────────

[AttributeUsage(AttributeTargets.Method)]
public class TestPriorityAttribute : Attribute
{
    public int Priority { get; }
    public TestPriorityAttribute(int priority) => Priority = priority;
}

public class PriorityOrderer : ITestCaseOrderer
{
    public IEnumerable<TTestCase> OrderTestCases<TTestCase>(IEnumerable<TTestCase> testCases)
        where TTestCase : ITestCase
    {
        var sorted = new SortedDictionary<int, List<TTestCase>>();

        foreach (var testCase in testCases)
        {
            var priority = testCase.TestMethod.Method
                .GetCustomAttributes(typeof(TestPriorityAttribute).AssemblyQualifiedName!)
                .FirstOrDefault()
                ?.GetNamedArgument<int>("Priority") ?? int.MaxValue;

            if (!sorted.TryGetValue(priority, out var list))
            {
                list = new List<TTestCase>();
                sorted[priority] = list;
            }
            list.Add(testCase);
        }

        return sorted.SelectMany(kvp => kvp.Value);
    }
}

// ── Test logger that writes to ITestOutputHelper ──────────────────

public class TestOutputLogger<T> : ILogger<T>
{
    private readonly ITestOutputHelper _output;

    public TestOutputLogger(ITestOutputHelper output) => _output = output;

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
    public bool IsEnabled(LogLevel logLevel) => true;

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        try
        {
            _output.WriteLine($"[{logLevel}] {formatter(state, exception)}");
        }
        catch
        {
            // ITestOutputHelper may throw if test has already completed
        }
    }
}
