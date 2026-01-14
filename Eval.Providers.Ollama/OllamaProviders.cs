using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Eval.Core;

namespace Eval.Providers.Ollama;

public class OllamaEmbeddingProvider : IEmbeddingProvider
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;

    public OllamaEmbeddingProvider(string baseUrl, HttpClient? httpClient = null)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _httpClient = httpClient ?? new HttpClient();
    }

    public async Task<float[]> EmbedAsync(string model, string text, CancellationToken cancellationToken)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/api/embeddings");
        var payload = JsonSerializer.Serialize(new { model, prompt = text });
        request.Content = new StringContent(payload, Encoding.UTF8, "application/json");

        using var response = await _httpClient.SendAsync(request, cancellationToken);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
        if (!doc.RootElement.TryGetProperty("embedding", out var embeddingElement))
        {
            throw new InvalidOperationException("Ollama embeddings response missing 'embedding'.");
        }

        var embedding = new List<float>();
        foreach (var number in embeddingElement.EnumerateArray())
        {
            embedding.Add(number.GetSingle());
        }

        return embedding.ToArray();
    }
}

public class OllamaJsonReranker : IReranker
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;

    public OllamaJsonReranker(string baseUrl, HttpClient? httpClient = null)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _httpClient = httpClient ?? new HttpClient
        {
            Timeout = Timeout.InfiniteTimeSpan
        };
    }

    public async Task<RerankResult> RerankAsync(string question, IReadOnlyList<RerankCandidate> candidates, RerankerConfig config, CancellationToken cancellationToken)
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cts.CancelAfter(TimeSpan.FromMilliseconds(config.TimeoutMs));

        var messages = BuildMessages(question, candidates, config.Prompt);
        var requestPayload = new
        {
            model = config.Model,
            stream = false,
            messages
        };

        var request = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/api/chat");
        var payload = JsonSerializer.Serialize(requestPayload);
        request.Content = new StringContent(payload, Encoding.UTF8, "application/json");

        using var response = await _httpClient.SendAsync(request, cts.Token);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync(cts.Token);
        var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cts.Token);
        var content = doc.RootElement.GetProperty("message").GetProperty("content").GetString() ?? string.Empty;

        var parsed = ParseJson(content);
        if (parsed == null || parsed.RankedChunkIds.Count == 0)
        {
            return new RerankResult(Array.Empty<string>(), true);
        }

        return new RerankResult(parsed.RankedChunkIds, false);
    }

    private static object[] BuildMessages(string question, IReadOnlyList<RerankCandidate> candidates, RerankerPromptConfig prompt)
    {
        var chunksFormatted = new StringBuilder();
        foreach (var candidate in candidates)
        {
            chunksFormatted.AppendLine($"- chunkId: {candidate.ChunkId}");
            chunksFormatted.AppendLine($"  text: \"{candidate.Text.Replace("\"", "\\\"")}\"");
        }

        var userPrompt = string.Join(Environment.NewLine, prompt.UserTemplate)
            .Replace("{{question}}", question)
            .Replace("{{chunks}}", chunksFormatted.ToString().TrimEnd());

        var messages = new List<object>
        {
            new { role = "system", content = prompt.System },
            new { role = "user", content = userPrompt }
        };

        return messages.ToArray();
    }

    private static RerankResponse? ParseJson(string content)
    {
        try
        {
            var parsed = JsonSerializer.Deserialize<RerankResponse>(content, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });
            return parsed;
        }
        catch
        {
            return null;
        }
    }

    private class RerankResponse
    {
        [JsonPropertyName("rankedChunkIds")]
        public List<string> RankedChunkIds { get; set; } = new();
    }
}
