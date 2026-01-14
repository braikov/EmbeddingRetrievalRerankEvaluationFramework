using System.Threading;
using System.Threading.Tasks;

namespace Eval.Core;

public interface IEmbeddingProvider
{
    Task<float[]> EmbedAsync(string model, string text, CancellationToken cancellationToken);
}

public interface IVectorIndex
{
    void Add(string id, float[] vector);
    IReadOnlyList<SearchResult> Search(float[] query, int topK);
    long EstimateSizeBytes();
    int Dimension { get; }
}

public interface IReranker
{
    Task<RerankResult> RerankAsync(string question, IReadOnlyList<RerankCandidate> candidates, RerankerConfig config, CancellationToken cancellationToken);
}
