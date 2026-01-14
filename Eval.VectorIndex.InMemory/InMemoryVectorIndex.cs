using Eval.Core;
using System.Diagnostics;

namespace Eval.VectorIndex.InMemory;

public class InMemoryVectorIndex : IVectorIndex
{
    private readonly List<string> _ids = new();
    private readonly List<float[]> _vectors = new();
    private int _dimension;

    public int Dimension => _dimension;

    public void Add(string id, float[] vector)
    {
        if (_dimension == 0)
        {
            _dimension = vector.Length;
        }
        else if (vector.Length != _dimension)
        {
            throw new InvalidOperationException($"Vector dimension mismatch. Expected {_dimension}, got {vector.Length}");
        }

        var normalized = Normalize(vector);
        _ids.Add(id);
        _vectors.Add(normalized);
    }

    public IReadOnlyList<SearchResult> Search(float[] query, int topK)
    {
        if (_dimension == 0)
        {
            return Array.Empty<SearchResult>();
        }

        var normalizedQuery = Normalize(query);
        var scores = new List<SearchResult>(_ids.Count);

        for (int i = 0; i < _ids.Count; i++)
        {
            var score = Cosine(normalizedQuery, _vectors[i]);
            scores.Add(new SearchResult(_ids[i], score));
        }

        return scores
            .OrderByDescending(s => s.Score)
            .Take(topK)
            .ToList();
    }

    public long EstimateSizeBytes()
    {
        if (_dimension == 0) return 0;
        long vectorBytes = (long)_ids.Count * _dimension * sizeof(float);
        return vectorBytes;
    }

    private static float[] Normalize(float[] vector)
    {
        var norm = Math.Sqrt(vector.Sum(v => v * v));
        if (norm == 0) return vector;
        return vector.Select(v => (float)(v / norm)).ToArray();
    }

    private static float Cosine(float[] a, float[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }
        return (float)sum;
    }
}
