// Copyright (c) Microsoft Corporation. All rights reserved.
// GoodMemOptions.cs

using System.ComponentModel.DataAnnotations;

namespace Microsoft.AutoGen.Extensions.GoodMem;

/// <summary>
/// Configuration options for connecting to a GoodMem API server.
/// Bind to the "GoodMem" configuration section.
/// </summary>
public class GoodMemOptions
{
    /// <summary>
    /// The base URL of the GoodMem API server (e.g., "http://localhost:8080").
    /// </summary>
    [Required]
    public required string BaseUrl { get; set; }

    /// <summary>
    /// The API key for authentication (sent as X-API-Key header).
    /// </summary>
    [Required]
    public required string ApiKey { get; set; }

    /// <summary>
    /// Name of the space to use. If it already exists, it will be reused.
    /// </summary>
    [Required]
    public required string SpaceName { get; set; }

    /// <summary>
    /// The embedder model ID for converting text to vector embeddings.
    /// Use <see cref="GoodMemClient.ListEmbeddersAsync"/> to discover available embedders.
    /// </summary>
    [Required]
    public required string EmbedderId { get; set; }

    /// <summary>
    /// Maximum number of results to return when querying memories.
    /// </summary>
    public int MaxResults { get; set; } = 5;

    /// <summary>
    /// Whether to fetch full memory metadata alongside matched chunks during retrieval.
    /// </summary>
    public bool IncludeMemoryDefinition { get; set; } = true;

    /// <summary>
    /// Retry for up to 60 seconds when no results are found during retrieval.
    /// Enable when memories were just added and may still be undergoing chunking and embedding.
    /// </summary>
    public bool WaitForIndexing { get; set; } = true;

    /// <summary>
    /// HTTP request timeout in seconds.
    /// </summary>
    public int TimeoutSeconds { get; set; } = 120;

    /// <summary>
    /// Whether to skip SSL certificate validation. Set to true for self-signed certificates.
    /// </summary>
    public bool SkipSslValidation { get; set; }
}

/// <summary>
/// Configuration for how documents are chunked before embedding.
/// Mirrors the reference integration's advanced chunking options.
/// </summary>
public class ChunkingOptions
{
    /// <summary>
    /// Number of characters per chunk when splitting documents.
    /// </summary>
    public int ChunkSize { get; set; } = 256;

    /// <summary>
    /// Number of overlapping characters between consecutive chunks.
    /// </summary>
    public int ChunkOverlap { get; set; } = 25;

    /// <summary>
    /// Where to attach the separator when splitting. One of: KEEP_END, KEEP_START, DISCARD.
    /// </summary>
    public string KeepStrategy { get; set; } = "KEEP_END";

    /// <summary>
    /// How chunk size is measured. One of: CHARACTER_COUNT, TOKEN_COUNT.
    /// </summary>
    public string LengthMeasurement { get; set; } = "CHARACTER_COUNT";

    /// <summary>
    /// Separators used for recursive splitting.
    /// </summary>
    public string[] Separators { get; set; } = new[] { "\n\n", "\n", ". ", " ", "" };

    /// <summary>
    /// Whether separators are regex patterns.
    /// </summary>
    public bool SeparatorIsRegex { get; set; }
}

/// <summary>
/// Configuration for retrieval post-processing (reranker and/or LLM).
/// Mirrors the reference integration's reranker/LLM options.
/// </summary>
public class PostProcessorOptions
{
    /// <summary>
    /// Optional reranker model ID to improve result ordering.
    /// </summary>
    public string? RerankerId { get; set; }

    /// <summary>
    /// Optional LLM ID to generate contextual responses alongside retrieved chunks.
    /// </summary>
    public string? LlmId { get; set; }

    /// <summary>
    /// Minimum score (0-1) for including results. Only used when Reranker or LLM is set.
    /// </summary>
    public double? RelevanceThreshold { get; set; }

    /// <summary>
    /// Creativity setting for LLM generation (0-2). Only used when LLM ID is set.
    /// </summary>
    public double? LlmTemperature { get; set; }

    /// <summary>
    /// Reorder results by creation time instead of relevance score.
    /// </summary>
    public bool ChronologicalResort { get; set; }
}
