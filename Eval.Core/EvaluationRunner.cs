using System.Diagnostics;

namespace Eval.Core;

public class EvaluationRunner
{
    private readonly EvaluationConfig _config;
    private readonly Dictionary<string, CorpusChunk> _corpus;
    private readonly List<Question> _questions;
    private readonly Dictionary<string, EmbeddingModelRuntime> _embeddingModels;
    private readonly Dictionary<string, RerankerConfig> _rerankerConfigs;
    private readonly Dictionary<string, IReranker> _rerankerImplementations;

    public EvaluationRunner(
        EvaluationConfig config,
        Dictionary<string, CorpusChunk> corpus,
        List<Question> questions,
        Dictionary<string, EmbeddingModelRuntime> embeddingModels,
        Dictionary<string, RerankerConfig> rerankerConfigs,
        Dictionary<string, IReranker> rerankerImplementations)
    {
        _config = config;
        _corpus = corpus;
        _questions = questions;
        _embeddingModels = embeddingModels;
        _rerankerConfigs = rerankerConfigs;
        _rerankerImplementations = rerankerImplementations;
    }

    public async Task<(List<ExperimentSummaryRow> Summary, List<PerQuestionResult> PerQuestion)> RunAsync(CancellationToken cancellationToken)
    {
        var summaryRows = new List<ExperimentSummaryRow>();
        var allPerQuestion = new List<PerQuestionResult>();

        foreach (var experiment in _config.Experiments)
        {
            var perQuestionResults = new List<PerQuestionResult>();
            var runtimeStats = new ExperimentRuntimeStats();
            int rerankErrors = 0;

            foreach (var question in _questions)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var totalSw = Stopwatch.StartNew();

                double embeddingLatency = 0;
                double searchLatency = 0;
                double rerankLatency = 0;
                bool usedFallback = false;

                var aggregatedScores = new Dictionary<string, double>();

                foreach (var source in experiment.CandidateSources)
                {
                    if (!_embeddingModels.TryGetValue(source.EmbeddingModel, out var modelRuntime))
                    {
                        throw new InvalidOperationException($"Embedding model '{source.EmbeddingModel}' is not configured.");
                    }

                    var embedSw = Stopwatch.StartNew();
                    var questionEmbedding = await modelRuntime.Provider.EmbedAsync(modelRuntime.Definition.Model, question.Query, cancellationToken);
                    embedSw.Stop();
                    embeddingLatency += embedSw.Elapsed.TotalMilliseconds;

                    var searchSw = Stopwatch.StartNew();
                    var results = modelRuntime.Index.Search(questionEmbedding, source.FetchK);
                    searchSw.Stop();
                    searchLatency += searchSw.Elapsed.TotalMilliseconds;

                    foreach (var result in results)
                    {
                        if (aggregatedScores.TryGetValue(result.ChunkId, out var current))
                        {
                            aggregatedScores[result.ChunkId] = AggregateScore(experiment.Merge?.AggregateScore, current, result.Score);
                        }
                        else
                        {
                            aggregatedScores[result.ChunkId] = result.Score;
                        }
                    }
                }

                var baselineRanked = aggregatedScores
                    .OrderByDescending(kv => kv.Value)
                    .Select(kv => new SearchResult(kv.Key, kv.Value))
                    .ToList();

                List<SearchResult> finalRanked;

                if (experiment.Type.Equals("rerank", StringComparison.OrdinalIgnoreCase))
                {
                    var rerankerConfig = ResolveReranker(experiment);
                    var reranker = _rerankerImplementations[rerankerConfig.Provider];
                    var candidateLimit = Math.Min(rerankerConfig.MaxCandidates, baselineRanked.Count);
                    if (candidateLimit == 0)
                    {
                        finalRanked = baselineRanked.Take(experiment.FinalTopK).ToList();
                        rerankErrors++;
                        totalSw.Stop();
                        perQuestionResults.Add(Metrics.ComputePerQuestionResult(
                            experiment.Id,
                            question,
                            finalRanked,
                            experiment.FinalTopK,
                            _config.Metrics.MrrK,
                            _config.Metrics.NdcgK,
                            true));
                        runtimeStats.QuestionEmbeddingLatenciesMs.Add(embeddingLatency);
                        runtimeStats.SearchLatenciesMs.Add(searchLatency);
                        runtimeStats.RerankLatenciesMs.Add(rerankLatency);
                        runtimeStats.TotalLatenciesMs.Add(totalSw.Elapsed.TotalMilliseconds);
                        continue;
                    }
                    var rerankCandidates = baselineRanked
                        .Take(candidateLimit)
                        .Select(r => new RerankCandidate(r.ChunkId, _corpus[r.ChunkId].Text, r.Score))
                        .ToList();

                    List<string> rerankedIds;

                    try
                    {
                        var rerankSw = Stopwatch.StartNew();
                        var rerankResult = await reranker.RerankAsync(question.Query, rerankCandidates, rerankerConfig, cancellationToken);
                        rerankSw.Stop();
                        rerankLatency += rerankSw.Elapsed.TotalMilliseconds;
                        rerankedIds = ValidateRerankResult(rerankResult.RankedChunkIds, rerankCandidates.Select(c => c.ChunkId).ToHashSet());
                        usedFallback = rerankResult.UsedFallback;
                    }
                    catch
                    {
                        rerankedIds = new List<string>();
                        usedFallback = true;
                    }

                    if (usedFallback && rerankerConfig.FallbackToBaselineOnError)
                    {
                        rerankedIds = baselineRanked.Select(r => r.ChunkId).ToList();
                    }

                    finalRanked = Reorder(baselineRanked, rerankedIds, experiment.FinalTopK);
                    if (usedFallback) rerankErrors++;
                }
                else
                {
                    finalRanked = baselineRanked.Take(experiment.FinalTopK).ToList();
                }

                totalSw.Stop();

                runtimeStats.QuestionEmbeddingLatenciesMs.Add(embeddingLatency);
                runtimeStats.SearchLatenciesMs.Add(searchLatency);
                runtimeStats.RerankLatenciesMs.Add(rerankLatency);
                runtimeStats.TotalLatenciesMs.Add(totalSw.Elapsed.TotalMilliseconds);

                var perQuestion = Metrics.ComputePerQuestionResult(
                    experiment.Id,
                    question,
                    finalRanked,
                    experiment.FinalTopK,
                    _config.Metrics.MrrK,
                    _config.Metrics.NdcgK,
                    usedFallback);

                perQuestionResults.Add(perQuestion);
            }

            var recalls = Metrics.RecallAtK(perQuestionResults, _config.Metrics.TopKs);
            var mrr = Metrics.MrrAtK(perQuestionResults);
            var ndcg = Metrics.NdcgAtK(perQuestionResults);
            var stability = Metrics.StabilityAtK(perQuestionResults, _config.Metrics.StabilityK);
            var questionCount = Math.Max(1, perQuestionResults.Count);

            var uniqueModels = experiment.CandidateSources.Select(cs => cs.EmbeddingModel).Distinct().ToList();
            var indexBuildTime = uniqueModels.Sum(m => _embeddingModels[m].IndexStats.BuildTimeMs);
            var indexSize = uniqueModels.Sum(m => _embeddingModels[m].IndexStats.SizeBytes);

            var summaryRow = new ExperimentSummaryRow(
                experiment.Id,
                experiment.Type,
                string.Join(" | ", experiment.CandidateSources.Select(cs => $"{cs.EmbeddingModel}@{cs.FetchK}")),
                experiment.RerankerId ?? string.Empty,
                experiment.FinalTopK,
                recalls,
                mrr,
                ndcg,
                stability,
                runtimeStats.Average(runtimeStats.QuestionEmbeddingLatenciesMs),
                runtimeStats.Average(runtimeStats.SearchLatenciesMs),
                runtimeStats.Average(runtimeStats.RerankLatenciesMs),
                runtimeStats.Average(runtimeStats.TotalLatenciesMs),
                indexBuildTime,
                indexSize,
                experiment.Type.Equals("rerank", StringComparison.OrdinalIgnoreCase) ? (double)rerankErrors / questionCount : 0
            );

            summaryRows.Add(summaryRow);
            allPerQuestion.AddRange(perQuestionResults);
        }

        return (summaryRows, allPerQuestion);
    }

    private static double AggregateScore(string? aggregateStrategy, double current, double incoming) =>
        (aggregateStrategy ?? "max").Equals("sum", StringComparison.OrdinalIgnoreCase)
            ? current + incoming
            : Math.Max(current, incoming);

    private RerankerConfig ResolveReranker(ExperimentConfig experiment)
    {
        if (experiment.RerankerId is null)
        {
            throw new InvalidOperationException($"Experiment '{experiment.Id}' is missing rerankerId.");
        }

        if (!_rerankerConfigs.TryGetValue(experiment.RerankerId, out var config))
        {
            throw new InvalidOperationException($"Reranker '{experiment.RerankerId}' is not configured.");
        }

        if (!_rerankerImplementations.ContainsKey(config.Provider))
        {
            throw new InvalidOperationException($"Reranker provider '{config.Provider}' is not available.");
        }

        return config;
    }

    private static List<string> ValidateRerankResult(IEnumerable<string> rankedChunkIds, HashSet<string> candidateIds)
    {
        var seen = new HashSet<string>();
        var clean = new List<string>();

        foreach (var id in rankedChunkIds)
        {
            if (!candidateIds.Contains(id)) continue;
            if (seen.Add(id))
            {
                clean.Add(id);
            }
        }

        return clean;
    }

    private static List<SearchResult> Reorder(List<SearchResult> baseline, List<string> rerankedIds, int finalTopK)
    {
        var byId = baseline.ToDictionary(r => r.ChunkId, r => r.Score);
        var ordered = new List<SearchResult>();
        var rerankSet = rerankedIds.ToHashSet();

        foreach (var id in rerankedIds)
        {
            if (byId.TryGetValue(id, out var score))
            {
                ordered.Add(new SearchResult(id, score));
            }
        }

        foreach (var result in baseline)
        {
            if (ordered.Count >= finalTopK) break;
            if (rerankSet.Contains(result.ChunkId)) continue;
            ordered.Add(result);
        }

        return ordered.Take(finalTopK).ToList();
    }
}
