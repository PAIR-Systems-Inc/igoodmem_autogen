// Copyright (c) Microsoft Corporation. All rights reserved.
// GoodMemHostingExtensions.cs

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace Microsoft.AutoGen.Extensions.GoodMem;

/// <summary>
/// Extension methods for registering GoodMem services with the host builder.
/// Follows the same pattern as SemanticKernelHostingExtensions.
/// </summary>
public static class GoodMemHostingExtensions
{
    /// <summary>
    /// Configures GoodMem services using the "GoodMem" configuration section.
    /// Registers GoodMemOptions, ChunkingOptions, PostProcessorOptions, GoodMemClient,
    /// and GoodMemMiddleware in the DI container.
    /// </summary>
    /// <param name="builder">The host application builder.</param>
    /// <returns>The builder for chaining.</returns>
    public static IHostApplicationBuilder ConfigureGoodMem(this IHostApplicationBuilder builder)
    {
        builder.Services.AddOptions<GoodMemOptions>()
            .Bind(builder.Configuration.GetSection("GoodMem"))
            .ValidateDataAnnotations()
            .ValidateOnStart();

        builder.Services.AddOptions<ChunkingOptions>()
            .Bind(builder.Configuration.GetSection("GoodMem:Chunking"));

        builder.Services.AddOptions<PostProcessorOptions>()
            .Bind(builder.Configuration.GetSection("GoodMem:PostProcessor"));

        builder.Services.AddSingleton<GoodMemClient>();
        builder.Services.AddTransient<GoodMemMiddleware>();

        return builder;
    }

    /// <summary>
    /// Configures GoodMem services with explicit options (no configuration binding).
    /// </summary>
    /// <param name="builder">The host application builder.</param>
    /// <param name="configureOptions">Action to configure GoodMem options.</param>
    /// <returns>The builder for chaining.</returns>
    public static IHostApplicationBuilder ConfigureGoodMem(
        this IHostApplicationBuilder builder,
        Action<GoodMemOptions> configureOptions)
    {
        builder.Services.Configure(configureOptions);
        builder.Services.AddOptions<ChunkingOptions>();
        builder.Services.AddOptions<PostProcessorOptions>();
        builder.Services.AddSingleton<GoodMemClient>();
        builder.Services.AddTransient<GoodMemMiddleware>();

        return builder;
    }
}
