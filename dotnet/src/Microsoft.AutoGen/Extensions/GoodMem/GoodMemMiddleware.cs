// Copyright (c) Microsoft Corporation. All rights reserved.
// GoodMemMiddleware.cs

using AutoGen.Core;
using Microsoft.Extensions.Logging;

namespace Microsoft.AutoGen.Extensions.GoodMem;

/// <summary>
/// Middleware that injects relevant GoodMem memories into the conversation context
/// before each agent turn. Retrieved memories are prepended as a System-role TextMessage.
///
/// This mirrors the Python integration's update_context behavior: extract the query from
/// the last message, perform semantic retrieval, and inject matching results.
/// </summary>
public class GoodMemMiddleware : IMiddleware
{
    private readonly GoodMemClient _client;
    private readonly ILogger<GoodMemMiddleware> _logger;

    public string? Name => "GoodMemMiddleware";

    /// <summary>
    /// Creates a new GoodMemMiddleware.
    /// </summary>
    /// <param name="client">The GoodMem client to use for retrieval.</param>
    /// <param name="logger">Logger instance.</param>
    public GoodMemMiddleware(GoodMemClient client, ILogger<GoodMemMiddleware> logger)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Intercepts the agent's GenerateReply call. Extracts the query from the last
    /// message, retrieves relevant memories from GoodMem, and injects them as a
    /// System TextMessage at the beginning of the message list.
    /// </summary>
    public async Task<IMessage> InvokeAsync(
        MiddlewareContext context,
        IAgent agent,
        CancellationToken cancellationToken = default)
    {
        var messages = context.Messages.ToList();

        if (messages.Count > 0)
        {
            // Extract query from the last message
            var lastMessage = messages.Last();
            var queryText = (lastMessage as ICanGetTextContent)?.GetContent();

            if (!string.IsNullOrWhiteSpace(queryText))
            {
                try
                {
                    var result = await _client.RetrieveMemoriesAsync(queryText, cancellationToken: cancellationToken).ConfigureAwait(false);

                    if (result.Results.Count > 0)
                    {
                        var memoryStrings = result.Results
                            .Select((chunk, i) => $"{i + 1}. {chunk.ChunkText}")
                            .ToArray();

                        var memoryContext = "\nRelevant memory content:\n" + string.Join("\n", memoryStrings);

                        var memoryMessage = new TextMessage(Role.System, memoryContext, "GoodMem");

                        // Prepend memory message to the conversation
                        messages.Insert(0, memoryMessage);

                        _logger.LogDebug("Injected {Count} GoodMem memories into conversation", result.Results.Count);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to retrieve GoodMem memories, proceeding without memory context");
                }
            }
        }

        // Invoke the agent with the (possibly augmented) messages
        var newContext = new MiddlewareContext(messages, context.Options);
        return await agent.GenerateReplyAsync(newContext.Messages, newContext.Options, cancellationToken).ConfigureAwait(false);
    }
}
