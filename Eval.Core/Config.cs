using System.Text.Json.Serialization;

namespace Eval.Core;

public class ProviderConfig
{
    [JsonPropertyName("baseUrl")]
    public string BaseUrl { get; set; } = string.Empty;
}

public class EmbeddingModelConfig
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("provider")]
    public string Provider { get; set; } = string.Empty;
}

public class RerankerPromptConfig
{
    [JsonPropertyName("system")]
    public string System { get; set; } = string.Empty;

    [JsonPropertyName("userTemplate")]
    public List<string> UserTemplate { get; set; } = new();
}

public class RerankerConfig
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("type")]
    public string Type { get; set; } = string.Empty;

    [JsonPropertyName("provider")]
    public string Provider { get; set; } = string.Empty;

    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("prompt")]
    public RerankerPromptConfig Prompt { get; set; } = new();

    [JsonPropertyName("maxCandidates")]
    public int MaxCandidates { get; set; } = 50;

    [JsonPropertyName("timeoutMs")]
    public int TimeoutMs { get; set; } = 30000;

    [JsonPropertyName("fallbackToBaselineOnError")]
    public bool FallbackToBaselineOnError { get; set; } = true;
}

public class MetricsConfig
{
    [JsonPropertyName("topKs")]
    public List<int> TopKs { get; set; } = new() { 1, 3, 5, 10 };

    [JsonPropertyName("mrrK")]
    public int MrrK { get; set; } = 10;

    [JsonPropertyName("ndcgK")]
    public int NdcgK { get; set; } = 10;

    [JsonPropertyName("stabilityK")]
    public int StabilityK { get; set; } = 5;
}

public class CandidateSourceConfig
{
    [JsonPropertyName("embeddingModel")]
    public string EmbeddingModel { get; set; } = string.Empty;

    [JsonPropertyName("fetchK")]
    public int FetchK { get; set; }
}

public class MergeConfig
{
    [JsonPropertyName("dedupeBy")]
    public string DedupeBy { get; set; } = "chunkId";

    [JsonPropertyName("aggregateScore")]
    public string AggregateScore { get; set; } = "max";
}

public class ExperimentConfig
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("type")]
    public string Type { get; set; } = string.Empty; // baseline | rerank

    [JsonPropertyName("candidateSources")]
    public List<CandidateSourceConfig> CandidateSources { get; set; } = new();

    [JsonPropertyName("merge")]
    public MergeConfig? Merge { get; set; }

    [JsonPropertyName("rerankerId")]
    public string? RerankerId { get; set; }

    [JsonPropertyName("finalTopK")]
    public int FinalTopK { get; set; } = 10;
}

public class EvaluationConfig
{
    [JsonPropertyName("providers")]
    public Dictionary<string, ProviderConfig> Providers { get; set; } = new();

    [JsonPropertyName("embeddingModels")]
    public List<EmbeddingModelConfig> EmbeddingModels { get; set; } = new();

    [JsonPropertyName("rerankers")]
    public List<RerankerConfig> Rerankers { get; set; } = new();

    [JsonPropertyName("metrics")]
    public MetricsConfig Metrics { get; set; } = new();

    [JsonPropertyName("experiments")]
    public List<ExperimentConfig> Experiments { get; set; } = new();
}
