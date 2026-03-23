// Copyright (c) Microsoft Corporation. All rights reserved.
// GoodMemClient.cs

using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace Microsoft.AutoGen.Extensions.GoodMem;

/// <summary>
/// HTTP client for the GoodMem API. Implements all core operations:
/// Create Space, Create Memory (text and file), Retrieve Memories,
/// Get Memory, Delete Memory, List Embedders, and List Spaces.
/// </summary>
public class GoodMemClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly GoodMemOptions _options;
    private readonly ChunkingOptions _chunking;
    private readonly PostProcessorOptions? _postProcessor;
    private readonly ILogger<GoodMemClient> _logger;
    private readonly bool _ownsHttpClient;
    private string? _spaceId;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    /// <summary>
    /// Creates a new GoodMemClient using the options pattern.
    /// </summary>
    public GoodMemClient(
        IOptions<GoodMemOptions> options,
        ILogger<GoodMemClient> logger,
        HttpClient? httpClient = null,
        IOptions<ChunkingOptions>? chunkingOptions = null,
        IOptions<PostProcessorOptions>? postProcessorOptions = null)
    {
        _options = options.Value;
        _chunking = chunkingOptions?.Value ?? new ChunkingOptions();
        _postProcessor = postProcessorOptions?.Value;
        _logger = logger;

        if (httpClient != null)
        {
            _httpClient = httpClient;
            _ownsHttpClient = false;
        }
        else
        {
            _httpClient = CreateHttpClient(_options);
            _ownsHttpClient = true;
        }
    }

    /// <summary>
    /// Creates a new GoodMemClient with explicit options (for standalone / test use).
    /// </summary>
    public GoodMemClient(
        GoodMemOptions options,
        ILogger<GoodMemClient> logger,
        HttpClient? httpClient = null,
        ChunkingOptions? chunking = null,
        PostProcessorOptions? postProcessor = null)
    {
        _options = options;
        _chunking = chunking ?? new ChunkingOptions();
        _postProcessor = postProcessor;
        _logger = logger;

        if (httpClient != null)
        {
            _httpClient = httpClient;
            _ownsHttpClient = false;
        }
        else
        {
            _httpClient = CreateHttpClient(_options);
            _ownsHttpClient = true;
        }
    }

    private static HttpClient CreateHttpClient(GoodMemOptions options)
    {
        HttpMessageHandler handler;
        if (options.SkipSslValidation)
        {
            handler = new HttpClientHandler
            {
                ServerCertificateCustomValidationCallback = (_, _, _, _) => true,
            };
        }
        else
        {
            handler = new HttpClientHandler();
        }

        return new HttpClient(handler) { Timeout = TimeSpan.FromSeconds(options.TimeoutSeconds) };
    }

    private string BaseUrl => _options.BaseUrl.TrimEnd('/');

    private HttpRequestMessage CreateRequest(HttpMethod method, string relativeUrl)
    {
        var request = new HttpRequestMessage(method, $"{BaseUrl}{relativeUrl}");
        request.Headers.Add("X-API-Key", _options.ApiKey);
        request.Headers.Add("Accept", "application/json");
        return request;
    }

    private HttpRequestMessage CreateJsonRequest<T>(HttpMethod method, string relativeUrl, T body)
    {
        var request = CreateRequest(method, relativeUrl);
        var json = JsonSerializer.Serialize(body, JsonOptions);
        request.Content = new StringContent(json, Encoding.UTF8, "application/json");
        return request;
    }

    // ── Space Operations ──────────────────────────────────────────

    /// <summary>
    /// Ensure the configured space exists, creating it if necessary.
    /// Returns the space ID. Caches the space ID after first lookup.
    /// Mirrors the reference integration's create-space logic: checks for existing
    /// space with same name before creating a new one, handles 409 Conflict.
    /// </summary>
    public async Task<string> EnsureSpaceAsync(CancellationToken cancellationToken = default)
    {
        if (_spaceId != null)
        {
            return _spaceId;
        }

        // Check if a space with the same name already exists
        try
        {
            var spaces = await ListSpacesAsync(cancellationToken).ConfigureAwait(false);
            var existing = spaces.FirstOrDefault(s =>
                string.Equals(GetStringProp(s, "name"), _options.SpaceName, StringComparison.Ordinal));

            if (existing != null)
            {
                _spaceId = GetStringProp(existing, "spaceId") ?? GetStringProp(existing, "id");
                _logger.LogInformation("Reusing existing GoodMem space: {SpaceName} ({SpaceId})", _options.SpaceName, _spaceId);
                return _spaceId!;
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to list spaces, will attempt to create");
        }

        // Create new space
        var result = await CreateSpaceAsync(_options.SpaceName, _options.EmbedderId, cancellationToken).ConfigureAwait(false);
        _spaceId = GetStringProp(result, "spaceId") ?? GetStringProp(result, "id");
        return _spaceId!;
    }

    /// <summary>
    /// Create a new space or reuse an existing one.
    /// Mirrors the reference integration's create-space action.
    /// </summary>
    public async Task<Dictionary<string, object?>> CreateSpaceAsync(
        string? name = null,
        string? embedderId = null,
        CancellationToken cancellationToken = default)
    {
        var spaceName = name ?? _options.SpaceName;
        var embedder = embedderId ?? _options.EmbedderId;

        // Check for existing space first
        try
        {
            var spaces = await ListSpacesAsync(cancellationToken).ConfigureAwait(false);
            var existing = spaces.FirstOrDefault(s =>
                string.Equals(GetStringProp(s, "name"), spaceName, StringComparison.Ordinal));

            if (existing != null)
            {
                var sid = GetStringProp(existing, "spaceId") ?? GetStringProp(existing, "id");
                if (spaceName == _options.SpaceName)
                {
                    _spaceId = sid;
                }

                return new Dictionary<string, object?>
                {
                    ["spaceId"] = sid,
                    ["name"] = spaceName,
                    ["embedderId"] = embedder,
                    ["reused"] = true,
                    ["message"] = "Space already exists, reusing existing space",
                };
            }
        }
        catch
        {
            // If listing fails, proceed to create
        }

        var requestBody = new Dictionary<string, object>
        {
            ["name"] = spaceName,
            ["spaceEmbedders"] = new[]
            {
                new Dictionary<string, object>
                {
                    ["embedderId"] = embedder,
                    ["defaultRetrievalWeight"] = 1.0,
                }
            },
            ["defaultChunkingConfig"] = new Dictionary<string, object>
            {
                ["recursive"] = new Dictionary<string, object>
                {
                    ["chunkSize"] = _chunking.ChunkSize,
                    ["chunkOverlap"] = _chunking.ChunkOverlap,
                    ["separators"] = _chunking.Separators,
                    ["keepStrategy"] = _chunking.KeepStrategy,
                    ["separatorIsRegex"] = _chunking.SeparatorIsRegex,
                    ["lengthMeasurement"] = _chunking.LengthMeasurement,
                }
            },
        };

        using var request = CreateJsonRequest(HttpMethod.Post, "/v1/spaces", requestBody);
        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);

        // Handle 409 Conflict: space was created between our list check and create
        if (response.StatusCode == System.Net.HttpStatusCode.Conflict)
        {
            _logger.LogDebug("Space creation returned 409 Conflict, re-listing to find existing space");
            var spaces = await ListSpacesAsync(cancellationToken).ConfigureAwait(false);
            var existing = spaces.FirstOrDefault(s =>
                string.Equals(GetStringProp(s, "name"), spaceName, StringComparison.Ordinal));

            if (existing != null)
            {
                var sid = GetStringProp(existing, "spaceId") ?? GetStringProp(existing, "id");
                if (spaceName == _options.SpaceName)
                {
                    _spaceId = sid;
                }

                return new Dictionary<string, object?>
                {
                    ["spaceId"] = sid,
                    ["name"] = spaceName,
                    ["embedderId"] = embedder,
                    ["reused"] = true,
                    ["message"] = "Space already exists (409), reusing existing space",
                };
            }
        }

        response.EnsureSuccessStatusCode();
        var result = await DeserializeResponseAsync(response, cancellationToken).ConfigureAwait(false);
        var spaceId = GetStringProp(result, "spaceId") ?? GetStringProp(result, "id");

        if (spaceName == _options.SpaceName)
        {
            _spaceId = spaceId;
        }

        _logger.LogInformation("Created GoodMem space: {SpaceName} ({SpaceId})", spaceName, spaceId);

        return new Dictionary<string, object?>
        {
            ["spaceId"] = spaceId,
            ["name"] = GetStringProp(result, "name"),
            ["embedderId"] = embedder,
            ["reused"] = false,
            ["message"] = "Space created successfully",
        };
    }

    /// <summary>
    /// List all available spaces from the GoodMem API.
    /// </summary>
    public async Task<List<Dictionary<string, object?>>> ListSpacesAsync(CancellationToken cancellationToken = default)
    {
        using var request = CreateRequest(HttpMethod.Get, "/v1/spaces");
        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var body = await DeserializeResponseAsync(response, cancellationToken).ConfigureAwait(false);
        return ExtractList(body, "spaces");
    }

    // ── Memory Operations ─────────────────────────────────────────

    /// <summary>
    /// Create a text memory in the configured space.
    /// Mirrors the reference integration's create-memory action for text content.
    /// </summary>
    public async Task<Dictionary<string, object?>> CreateTextMemoryAsync(
        string textContent,
        Dictionary<string, object>? metadata = null,
        CancellationToken cancellationToken = default)
    {
        var spaceId = await EnsureSpaceAsync(cancellationToken).ConfigureAwait(false);

        var requestBody = new Dictionary<string, object>
        {
            ["spaceId"] = spaceId,
            ["contentType"] = "text/plain",
            ["originalContent"] = textContent,
        };

        if (metadata != null && metadata.Count > 0)
        {
            requestBody["metadata"] = metadata;
        }

        using var request = CreateJsonRequest(HttpMethod.Post, "/v1/memories", requestBody);
        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var result = await DeserializeResponseAsync(response, cancellationToken).ConfigureAwait(false);
        var memoryId = GetStringProp(result, "memoryId");
        _logger.LogInformation("Created GoodMem text memory: {MemoryId}", memoryId);

        return new Dictionary<string, object?>
        {
            ["memoryId"] = memoryId,
            ["spaceId"] = GetStringProp(result, "spaceId"),
            ["status"] = GetStringProp(result, "processingStatus") ?? "PENDING",
            ["contentType"] = "text/plain",
            ["message"] = "Memory created successfully",
        };
    }

    /// <summary>
    /// Create a file memory (PDF, DOCX, image, etc.) in the configured space.
    /// Auto-detects content type from file extension. Binary files are base64-encoded.
    /// Mirrors the reference integration's create-memory action for file content.
    /// </summary>
    public async Task<Dictionary<string, object?>> CreateFileMemoryAsync(
        string filePath,
        Dictionary<string, object>? metadata = null,
        CancellationToken cancellationToken = default)
    {
        var spaceId = await EnsureSpaceAsync(cancellationToken).ConfigureAwait(false);

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"File not found: {filePath}");
        }

        var extension = Path.GetExtension(filePath);
        var mimeType = GetMimeType(extension) ?? "application/octet-stream";

        var requestBody = new Dictionary<string, object>
        {
            ["spaceId"] = spaceId,
            ["contentType"] = mimeType,
        };

        var fileBytes = await File.ReadAllBytesAsync(filePath, cancellationToken).ConfigureAwait(false);

        if (mimeType.StartsWith("text/", StringComparison.OrdinalIgnoreCase))
        {
            // Text files: decode and send as originalContent
            requestBody["originalContent"] = Encoding.UTF8.GetString(fileBytes);
        }
        else
        {
            // Binary files (PDF, images, etc.): send as base64
            requestBody["originalContentB64"] = Convert.ToBase64String(fileBytes);
        }

        if (metadata != null && metadata.Count > 0)
        {
            requestBody["metadata"] = metadata;
        }

        using var request = CreateJsonRequest(HttpMethod.Post, "/v1/memories", requestBody);
        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var result = await DeserializeResponseAsync(response, cancellationToken).ConfigureAwait(false);
        var memoryId = GetStringProp(result, "memoryId");
        _logger.LogInformation("Created GoodMem file memory: {MemoryId} ({FileName})", memoryId, Path.GetFileName(filePath));

        return new Dictionary<string, object?>
        {
            ["memoryId"] = memoryId,
            ["spaceId"] = GetStringProp(result, "spaceId"),
            ["status"] = GetStringProp(result, "processingStatus") ?? "PENDING",
            ["contentType"] = mimeType,
            ["fileName"] = Path.GetFileName(filePath),
            ["message"] = "Memory created successfully",
        };
    }

    /// <summary>
    /// Retrieve semantically similar memories from one or more spaces.
    /// Supports wait-for-indexing polling, NDJSON/SSE response parsing,
    /// and optional post-processor configuration.
    /// Mirrors the reference integration's retrieve-memories action.
    /// </summary>
    public async Task<RetrieveResult> RetrieveMemoriesAsync(
        string query,
        string[]? spaceIds = null,
        int? maxResults = null,
        CancellationToken cancellationToken = default)
    {
        var defaultSpaceId = await EnsureSpaceAsync(cancellationToken).ConfigureAwait(false);
        var selectedSpaceIds = spaceIds ?? new[] { defaultSpaceId };
        var requestedSize = maxResults ?? _options.MaxResults;

        var spaceKeys = selectedSpaceIds.Select(id => new Dictionary<string, string> { ["spaceId"] = id }).ToArray();

        var requestBody = new Dictionary<string, object>
        {
            ["message"] = query,
            ["spaceKeys"] = spaceKeys,
            ["requestedSize"] = requestedSize,
            ["fetchMemory"] = _options.IncludeMemoryDefinition,
        };

        // Add post-processor config if configured
        if (_postProcessor != null && (_postProcessor.RerankerId != null || _postProcessor.LlmId != null))
        {
            var config = new Dictionary<string, object>();
            if (_postProcessor.RerankerId != null)
            {
                config["reranker_id"] = _postProcessor.RerankerId;
            }

            if (_postProcessor.LlmId != null)
            {
                config["llm_id"] = _postProcessor.LlmId;
            }

            if (_postProcessor.RelevanceThreshold.HasValue)
            {
                config["relevance_threshold"] = _postProcessor.RelevanceThreshold.Value;
            }

            if (_postProcessor.LlmTemperature.HasValue)
            {
                config["llm_temp"] = _postProcessor.LlmTemperature.Value;
            }

            config["max_results"] = requestedSize;
            if (_postProcessor.ChronologicalResort)
            {
                config["chronological_resort"] = true;
            }

            requestBody["postProcessor"] = new Dictionary<string, object>
            {
                ["name"] = "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory",
                ["config"] = config,
            };
        }

        // Retrieval with wait-for-indexing polling
        var maxWaitMs = 60_000;
        var pollIntervalMs = 5_000;
        var shouldWait = _options.WaitForIndexing;
        var startTime = Environment.TickCount64;
        RetrieveResult? lastResult = null;

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            using var request = CreateJsonRequest(HttpMethod.Post, "/v1/memories:retrieve", requestBody);
            request.Headers.Remove("Accept");
            request.Headers.Add("Accept", "application/x-ndjson");

            using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var responseText = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            lastResult = ParseRetrieveResponse(responseText, query);

            if (lastResult.Results.Count > 0 || !shouldWait)
            {
                return lastResult;
            }

            var elapsed = Environment.TickCount64 - startTime;
            if (elapsed >= maxWaitMs)
            {
                _logger.LogWarning("No results found after waiting 60 seconds for indexing");
                return lastResult;
            }

            await Task.Delay(pollIntervalMs, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Fetch a specific memory by its ID, with optional content.
    /// Mirrors the reference integration's get-memory action.
    /// </summary>
    public async Task<Dictionary<string, object?>> GetMemoryAsync(
        string memoryId,
        bool includeContent = true,
        CancellationToken cancellationToken = default)
    {
        using var request = CreateRequest(HttpMethod.Get, $"/v1/memories/{memoryId}");
        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var memory = await DeserializeResponseAsync(response, cancellationToken).ConfigureAwait(false);
        var result = new Dictionary<string, object?>
        {
            ["memory"] = memory,
        };

        if (includeContent)
        {
            try
            {
                using var contentRequest = CreateRequest(HttpMethod.Get, $"/v1/memories/{memoryId}/content");
                using var contentResponse = await _httpClient.SendAsync(contentRequest, cancellationToken).ConfigureAwait(false);
                contentResponse.EnsureSuccessStatusCode();
                result["content"] = await DeserializeResponseAsync(contentResponse, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                result["contentError"] = $"Failed to fetch content: {ex.Message}";
            }
        }

        return result;
    }

    /// <summary>
    /// Permanently delete a memory and its associated chunks and vector embeddings.
    /// Mirrors the reference integration's delete-memory action.
    /// </summary>
    public async Task<Dictionary<string, object?>> DeleteMemoryAsync(
        string memoryId,
        CancellationToken cancellationToken = default)
    {
        using var request = CreateRequest(HttpMethod.Delete, $"/v1/memories/{memoryId}");
        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        _logger.LogInformation("Deleted GoodMem memory: {MemoryId}", memoryId);
        return new Dictionary<string, object?>
        {
            ["memoryId"] = memoryId,
            ["success"] = true,
            ["message"] = "Memory deleted successfully",
        };
    }

    // ── Discovery Operations ──────────────────────────────────────

    /// <summary>
    /// List available embedder models from the GoodMem API.
    /// </summary>
    public async Task<List<Dictionary<string, object?>>> ListEmbeddersAsync(CancellationToken cancellationToken = default)
    {
        using var request = CreateRequest(HttpMethod.Get, "/v1/embedders");
        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var body = await DeserializeResponseAsync(response, cancellationToken).ConfigureAwait(false);
        return ExtractList(body, "embedders");
    }

    // ── Helpers ───────────────────────────────────────────────────

    /// <summary>
    /// Reset cached space ID (useful for testing or reconfiguration).
    /// </summary>
    public void ResetSpaceCache()
    {
        _spaceId = null;
    }

    private static async Task<Dictionary<string, object?>> DeserializeResponseAsync(
        HttpResponseMessage response,
        CancellationToken cancellationToken)
    {
        var json = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        return DeserializeJsonObject(json);
    }

    private static Dictionary<string, object?> DeserializeJsonObject(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return JsonElementToDict(doc.RootElement);
    }

    private static Dictionary<string, object?> JsonElementToDict(JsonElement element)
    {
        var dict = new Dictionary<string, object?>();
        if (element.ValueKind != JsonValueKind.Object)
        {
            return dict;
        }

        foreach (var prop in element.EnumerateObject())
        {
            dict[prop.Name] = JsonElementToObject(prop.Value);
        }
        return dict;
    }

    private static object? JsonElementToObject(JsonElement element)
    {
        return element.ValueKind switch
        {
            JsonValueKind.Object => JsonElementToDict(element),
            JsonValueKind.Array => element.EnumerateArray().Select(JsonElementToObject).ToList(),
            JsonValueKind.String => element.GetString(),
            JsonValueKind.Number => element.TryGetInt64(out var l) ? l : element.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Null => null,
            _ => element.GetRawText(),
        };
    }

    private static List<Dictionary<string, object?>> ExtractList(Dictionary<string, object?> body, string key)
    {
        // API may return an array directly or an object with the keyed list
        if (body.TryGetValue(key, out var val) && val is List<object?> list)
        {
            return list
                .OfType<Dictionary<string, object?>>()
                .ToList();
        }

        // If the body itself represents a single-element or is structured differently,
        // try to interpret the entire body as containing list items
        return new List<Dictionary<string, object?>>();
    }

    private static string? GetStringProp(Dictionary<string, object?> dict, string key)
    {
        return dict.TryGetValue(key, out var val) ? val?.ToString() : null;
    }

    private static RetrieveResult ParseRetrieveResponse(string responseText, string query)
    {
        var results = new List<RetrievedChunk>();
        var memories = new List<Dictionary<string, object?>>();
        string? resultSetId = null;
        Dictionary<string, object?>? abstractReply = null;

        var lines = responseText.Trim().Split('\n');
        foreach (var line in lines)
        {
            var jsonStr = line.Trim();
            if (string.IsNullOrEmpty(jsonStr))
            {
                continue;
            }

            // Handle SSE format: extract JSON from "data: {...}" lines
            if (jsonStr.StartsWith("data:", StringComparison.Ordinal))
            {
                jsonStr = jsonStr.Substring(5).Trim();
            }
            // Skip SSE event type lines
            if (jsonStr.StartsWith("event:", StringComparison.Ordinal) || string.IsNullOrEmpty(jsonStr))
            {
                continue;
            }

            try
            {
                using var doc = JsonDocument.Parse(jsonStr);
                var root = doc.RootElement;

                if (root.TryGetProperty("resultSetBoundary", out var boundary))
                {
                    if (boundary.TryGetProperty("resultSetId", out var rsId))
                    {
                        resultSetId = rsId.GetString();
                    }
                }
                else if (root.TryGetProperty("memoryDefinition", out var memDef))
                {
                    memories.Add(JsonElementToDict(memDef));
                }
                else if (root.TryGetProperty("abstractReply", out var absReply))
                {
                    abstractReply = JsonElementToDict(absReply);
                }
                else if (root.TryGetProperty("retrievedItem", out var retrievedItem))
                {
                    var chunk = new RetrievedChunk();

                    if (retrievedItem.TryGetProperty("chunk", out var chunkWrapper))
                    {
                        if (chunkWrapper.TryGetProperty("chunk", out var innerChunk))
                        {
                            if (innerChunk.TryGetProperty("chunkId", out var cid))
                            {
                                chunk.ChunkId = cid.GetString();
                            }

                            if (innerChunk.TryGetProperty("chunkText", out var ctext))
                            {
                                chunk.ChunkText = ctext.GetString();
                            }

                            if (innerChunk.TryGetProperty("memoryId", out var mid))
                            {
                                chunk.MemoryId = mid.GetString();
                            }
                        }

                        if (chunkWrapper.TryGetProperty("relevanceScore", out var score))
                        {
                            chunk.RelevanceScore = score.GetDouble();
                        }

                        if (chunkWrapper.TryGetProperty("memoryIndex", out var mIdx))
                        {
                            chunk.MemoryIndex = mIdx.TryGetInt64(out var idx) ? (int)idx : null;
                        }
                    }

                    results.Add(chunk);
                }
            }
            catch (JsonException)
            {
                // Skip non-JSON lines (SSE event types, close events)
            }
        }

        return new RetrieveResult
        {
            ResultSetId = resultSetId,
            Results = results,
            Memories = memories,
            AbstractReply = abstractReply,
            Query = query,
        };
    }

    private static readonly Dictionary<string, string> MimeTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        [".pdf"] = "application/pdf",
        [".png"] = "image/png",
        [".jpg"] = "image/jpeg",
        [".jpeg"] = "image/jpeg",
        [".gif"] = "image/gif",
        [".webp"] = "image/webp",
        [".txt"] = "text/plain",
        [".html"] = "text/html",
        [".md"] = "text/markdown",
        [".csv"] = "text/csv",
        [".json"] = "application/json",
        [".xml"] = "application/xml",
        [".doc"] = "application/msword",
        [".docx"] = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        [".xls"] = "application/vnd.ms-excel",
        [".xlsx"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        [".ppt"] = "application/vnd.ms-powerpoint",
        [".pptx"] = "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    };

    private static string? GetMimeType(string extension)
    {
        return MimeTypes.TryGetValue(extension, out var mimeType) ? mimeType : null;
    }

    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _httpClient.Dispose();
        }
    }
}

/// <summary>
/// Result of a memory retrieval operation.
/// </summary>
public class RetrieveResult
{
    public string? ResultSetId { get; set; }
    public List<RetrievedChunk> Results { get; set; } = new();
    public List<Dictionary<string, object?>> Memories { get; set; } = new();
    public Dictionary<string, object?>? AbstractReply { get; set; }
    public string? Query { get; set; }
    public int TotalResults => Results.Count;
}

/// <summary>
/// A single retrieved chunk from a memory retrieval.
/// </summary>
public class RetrievedChunk
{
    public string? ChunkId { get; set; }
    public string? ChunkText { get; set; }
    public string? MemoryId { get; set; }
    public double? RelevanceScore { get; set; }
    public int? MemoryIndex { get; set; }
}
