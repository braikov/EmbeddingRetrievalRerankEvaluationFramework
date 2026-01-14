using System.Configuration;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using Eval.Core;
using Eval.Providers.Ollama;
using Eval.VectorIndex.InMemory;

var options = LoadOptionsFromAppConfig();
if (options == null)
{
    PrintConfigHelp();
    return;
}

Directory.CreateDirectory(options.OutputDirectory);

var config = DataLoaders.LoadConfig(options.ConfigPath);
ValidateConfig(config);
var corpus = DataLoaders.LoadCorpus(options.CorpusPath);
var questions = DataLoaders.LoadQuestions(options.QuestionsPath);

var embeddingProviders = BuildEmbeddingProviders(config);
var rerankerProviders = BuildRerankers(config);

var embeddingModels = new Dictionary<string, EmbeddingModelRuntime>();
foreach (var modelConfig in config.EmbeddingModels)
{
    if (!embeddingProviders.TryGetValue(modelConfig.Provider, out var provider))
    {
        throw new InvalidOperationException($"Embedding provider '{modelConfig.Provider}' is not configured.");
    }

    var definition = new EmbeddingModelDefinition(modelConfig.Name, modelConfig.Provider, modelConfig.Name);
    var (index, stats) = await BuildIndexAsync(definition, provider, corpus, CancellationToken.None);
    embeddingModels[modelConfig.Name] = new EmbeddingModelRuntime(definition, provider, index, stats);
}

var rerankerConfigs = config.Rerankers.ToDictionary(r => r.Id, r => r);

var runner = new EvaluationRunner(
    config,
    corpus,
    questions,
    embeddingModels,
    rerankerConfigs,
    rerankerProviders);

var (summary, perQuestion) = await runner.RunAsync(CancellationToken.None);

WriteSummaryCsv(Path.Combine(options.OutputDirectory, "results.summary.csv"), summary, config.Metrics.TopKs, config.Metrics.MrrK, config.Metrics.NdcgK, config.Metrics.StabilityK);
WritePerQuestionCsv(Path.Combine(options.OutputDirectory, "results.per_question.csv"), perQuestion, options.Delimiter);

PrintConsoleSummary(summary, config.Metrics.TopKs);

static CliOptions? LoadOptionsFromAppConfig()
{
    var configPath = ConfigurationManager.AppSettings["ConfigPath"];
    var corpusPath = ConfigurationManager.AppSettings["CorpusPath"];
    var questionsPath = ConfigurationManager.AppSettings["QuestionsPath"];
    var output = ConfigurationManager.AppSettings["OutputDirectory"] ?? "results";
    var delimiter = ConfigurationManager.AppSettings["Delimiter"] ?? "|";

    if (string.IsNullOrWhiteSpace(configPath) ||
        string.IsNullOrWhiteSpace(corpusPath) ||
        string.IsNullOrWhiteSpace(questionsPath))
    {
        return null;
    }

    return new CliOptions(configPath, corpusPath, questionsPath, output, delimiter);
}

static void PrintConfigHelp()
{
    Console.WriteLine("Missing required settings in app.config (ConfigPath, CorpusPath, QuestionsPath).");
    Console.WriteLine("Populate Eval.App/App.config or copy from App.config.example.");
}

static void ValidateConfig(EvaluationConfig config)
{
    var embeddingModelNames = config.EmbeddingModels.Select(m => m.Name).ToHashSet(StringComparer.OrdinalIgnoreCase);
    var maxRecallK = config.Metrics.TopKs.Count > 0 ? config.Metrics.TopKs.Max() : config.Metrics.MrrK;

    foreach (var experiment in config.Experiments)
    {
        if (experiment.CandidateSources.Count == 0)
        {
            throw new InvalidOperationException($"Experiment '{experiment.Id}' has no candidateSources.");
        }

        foreach (var source in experiment.CandidateSources)
        {
            if (!embeddingModelNames.Contains(source.EmbeddingModel))
            {
                throw new InvalidOperationException($"Experiment '{experiment.Id}' references unknown embedding model '{source.EmbeddingModel}'.");
            }
        }

        if (experiment.Type.Equals("rerank", StringComparison.OrdinalIgnoreCase) && string.IsNullOrEmpty(experiment.RerankerId))
        {
            throw new InvalidOperationException($"Experiment '{experiment.Id}' is rerank type but has no rerankerId.");
        }

        if (experiment.FinalTopK < maxRecallK)
        {
            Console.WriteLine($"Warning: Experiment '{experiment.Id}' finalTopK ({experiment.FinalTopK}) is below max recall@K ({maxRecallK}). Metrics may be truncated.");
        }
    }
}

static Dictionary<string, IEmbeddingProvider> BuildEmbeddingProviders(EvaluationConfig config)
{
    var dict = new Dictionary<string, IEmbeddingProvider>();
    foreach (var provider in config.Providers)
    {
        switch (provider.Key.ToLowerInvariant())
        {
            case "ollama":
                dict[provider.Key] = new OllamaEmbeddingProvider(provider.Value.BaseUrl);
                break;
            default:
                throw new InvalidOperationException($"Unknown embedding provider '{provider.Key}'.");
        }
    }
    return dict;
}

static Dictionary<string, IReranker> BuildRerankers(EvaluationConfig config)
{
    var dict = new Dictionary<string, IReranker>();
    foreach (var provider in config.Providers)
    {
        switch (provider.Key.ToLowerInvariant())
        {
            case "ollama":
                dict[provider.Key] = new OllamaJsonReranker(provider.Value.BaseUrl);
                break;
            default:
                throw new InvalidOperationException($"Unknown reranker provider '{provider.Key}'.");
        }
    }
    return dict;
}

static async Task<(InMemoryVectorIndex Index, IndexBuildStats Stats)> BuildIndexAsync(
    EmbeddingModelDefinition definition,
    IEmbeddingProvider provider,
    Dictionary<string, CorpusChunk> corpus,
    CancellationToken cancellationToken)
{
    var index = new InMemoryVectorIndex();
    var sw = Stopwatch.StartNew();

    foreach (var chunk in corpus.Values)
    {
        var embedding = await provider.EmbedAsync(definition.Model, chunk.Text, cancellationToken);
        index.Add(chunk.ChunkId, embedding);
    }

    sw.Stop();
    var stats = new IndexBuildStats((long)sw.Elapsed.TotalMilliseconds, index.EstimateSizeBytes(), index.Dimension);
    return (index, stats);
}

static void WriteSummaryCsv(string path, List<ExperimentSummaryRow> rows, IReadOnlyList<int> topKs, int mrrK, int ndcgK, int stabilityK)
{
    using var writer = new StreamWriter(path, false, new UTF8Encoding(false));
    var recallColumns = string.Join(",", topKs.Select(k => $"Recall@{k}"));

    writer.WriteLine($"ExperimentId,ExperimentType,CandidateSources,Reranker,FinalTopK,{recallColumns},MRR@{mrrK},nDCG@{ndcgK},Stability@{stabilityK},AvgQuestionEmbeddingLatencyMs,AvgSearchLatencyMs,AvgRerankLatencyMs,AvgTotalLatencyMs,IndexBuildTimeMs,IndexSizeBytes,RerankErrorRate");

    foreach (var row in rows)
    {
        var recallValues = string.Join(",", topKs.Select(k => row.Recalls.TryGetValue(k, out var v) ? v.ToString("0.000", CultureInfo.InvariantCulture) : "0"));
        writer.WriteLine(string.Join(",",
            Escape(row.ExperimentId),
            Escape(row.ExperimentType),
            Escape(row.CandidateSources),
            Escape(row.Reranker),
            row.FinalTopK.ToString(CultureInfo.InvariantCulture),
            recallValues,
            row.Mrr.ToString("0.000", CultureInfo.InvariantCulture),
            row.Ndcg.ToString("0.000", CultureInfo.InvariantCulture),
            row.Stability.ToString("0.000", CultureInfo.InvariantCulture),
            row.AvgQuestionEmbeddingLatencyMs.ToString("0.0", CultureInfo.InvariantCulture),
            row.AvgSearchLatencyMs.ToString("0.0", CultureInfo.InvariantCulture),
            row.AvgRerankLatencyMs.ToString("0.0", CultureInfo.InvariantCulture),
            row.AvgTotalLatencyMs.ToString("0.0", CultureInfo.InvariantCulture),
            row.IndexBuildTimeMs.ToString(CultureInfo.InvariantCulture),
            row.IndexSizeBytes.ToString(CultureInfo.InvariantCulture),
            row.RerankErrorRate.ToString("0.000", CultureInfo.InvariantCulture)
        ));
    }
}

static void WritePerQuestionCsv(string path, List<PerQuestionResult> rows, string delimiter)
{
    using var writer = new StreamWriter(path, false, new UTF8Encoding(false));
    writer.WriteLine("ExperimentId,QuestionId,GroupId,VariantId,Hit@K,RankFirstRelevant,MRRContribution,nDCG@K,TopKChunkIds,TopKScores,UsedFallback");

    foreach (var row in rows)
    {
        var chunkIds = string.Join(delimiter, row.TopKChunkIds);
        var scores = string.Join(delimiter, row.TopKScores.Select(s => s.ToString("0.000", CultureInfo.InvariantCulture)));

        writer.WriteLine(string.Join(",",
            Escape(row.ExperimentId),
            Escape(row.QuestionId),
            Escape(row.GroupId),
            Escape(row.VariantId),
            row.HitAtK ? "1" : "0",
            row.RankFirstRelevant.ToString(CultureInfo.InvariantCulture),
            row.MrrContribution.ToString("0.000", CultureInfo.InvariantCulture),
            row.NdcgAtK.ToString("0.000", CultureInfo.InvariantCulture),
            Escape(chunkIds),
            Escape(scores),
            row.UsedFallback ? "1" : "0"
        ));
    }
}

static void PrintConsoleSummary(List<ExperimentSummaryRow> rows, IReadOnlyList<int> topKs)
{
    Console.WriteLine("Experiment Results");
    foreach (var row in rows)
    {
        var recalls = string.Join(", ", topKs.Select(k => $"R@{k}={row.Recalls.GetValueOrDefault(k):0.000}"));
        Console.WriteLine($"{row.ExperimentId} ({row.ExperimentType}) | {recalls} | MRR={row.Mrr:0.000} | nDCG={row.Ndcg:0.000} | Stability={row.Stability:0.000}");
    }
}

static string Escape(string value)
{
    if (value.Contains(','))
    {
        return $"\"{value.Replace("\"", "\"\"")}\"";
    }
    return value;
}

record CliOptions(string ConfigPath, string CorpusPath, string QuestionsPath, string OutputDirectory, string Delimiter);
