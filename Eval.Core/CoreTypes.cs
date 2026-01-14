namespace Eval.Core;

public record CorpusChunk(string ChunkId, string SourceId, string Text);

public record Question
(
    string QuestionId,
    string GroupId,
    string VariantId,
    string Query,
    IReadOnlyList<string> RelevantChunkIds,
    IReadOnlyDictionary<string, int> RelevanceGrades
);

public record EmbeddingModelDefinition(string Name, string Provider, string Model);

public record SearchResult(string ChunkId, double Score);

public record RerankCandidate(string ChunkId, string Text, double Score);

public record RerankResult(IReadOnlyList<string> RankedChunkIds, bool UsedFallback);

public record IndexBuildStats(long BuildTimeMs, long SizeBytes, int Dimension);

public record EmbeddingModelRuntime(EmbeddingModelDefinition Definition, IEmbeddingProvider Provider, IVectorIndex Index, IndexBuildStats IndexStats);

public record PerQuestionResult(
    string ExperimentId,
    string QuestionId,
    string GroupId,
    string VariantId,
    bool HitAtK,
    int RankFirstRelevant,
    double MrrContribution,
    double NdcgAtK,
    IReadOnlyList<string> TopKChunkIds,
    IReadOnlyList<double> TopKScores,
    bool UsedFallback
);

public record ExperimentSummaryRow(
    string ExperimentId,
    string ExperimentType,
    string CandidateSources,
    string Reranker,
    int FinalTopK,
    Dictionary<int, double> Recalls,
    double Mrr,
    double Ndcg,
    double Stability,
    double AvgQuestionEmbeddingLatencyMs,
    double AvgSearchLatencyMs,
    double AvgRerankLatencyMs,
    double AvgTotalLatencyMs,
    long IndexBuildTimeMs,
    long IndexSizeBytes,
    double RerankErrorRate
);

public class ExperimentRuntimeStats
{
    public List<double> QuestionEmbeddingLatenciesMs { get; } = new();
    public List<double> SearchLatenciesMs { get; } = new();
    public List<double> RerankLatenciesMs { get; } = new();
    public List<double> TotalLatenciesMs { get; } = new();

    public double Average(List<double> values) => values.Count == 0 ? 0 : values.Average();
}
