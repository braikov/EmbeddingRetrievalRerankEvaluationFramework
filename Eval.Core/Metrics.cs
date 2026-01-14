namespace Eval.Core;

public static class Metrics
{
    public static Dictionary<int, double> RecallAtK(IEnumerable<PerQuestionResult> perQuestion, IReadOnlyList<int> topKs)
    {
        var recalls = new Dictionary<int, double>();
        foreach (var k in topKs)
        {
            var hitCount = perQuestion.Count(r => r.RankFirstRelevant > 0 && r.RankFirstRelevant <= k);
            var total = perQuestion.Count();
            recalls[k] = total == 0 ? 0 : (double)hitCount / total;
        }
        return recalls;
    }

    public static double MrrAtK(IEnumerable<PerQuestionResult> perQuestion)
    {
        var values = perQuestion.Select(r => r.MrrContribution).ToList();
        if (values.Count == 0) return 0;
        return values.Average();
    }

    public static double NdcgAtK(IEnumerable<PerQuestionResult> perQuestion)
    {
        var values = perQuestion.Select(r => r.NdcgAtK).ToList();
        if (values.Count == 0) return 0;
        return values.Average();
    }

    public static double StabilityAtK(IEnumerable<PerQuestionResult> perQuestion, int stabilityK)
    {
        var grouped = perQuestion.GroupBy(r => r.GroupId);
        var groupScores = new List<double>();

        foreach (var group in grouped)
        {
            var variantResults = group.ToList();
            if (variantResults.Count == 1)
            {
                groupScores.Add(1.0);
                continue;
            }

            var sets = variantResults
                .Select(r => r.TopKChunkIds.Take(stabilityK).ToHashSet())
                .ToList();

            var pairwise = new List<double>();
            for (int i = 0; i < sets.Count; i++)
            {
                for (int j = i + 1; j < sets.Count; j++)
                {
                    var intersection = sets[i].Intersect(sets[j]).Count();
                    var union = sets[i].Union(sets[j]).Count();
                    pairwise.Add(union == 0 ? 0 : (double)intersection / union);
                }
            }

            if (pairwise.Count > 0)
            {
                groupScores.Add(pairwise.Average());
            }
        }

        return groupScores.Count == 0 ? 0 : groupScores.Average();
    }

    public static PerQuestionResult ComputePerQuestionResult(
        string experimentId,
        Question question,
        IReadOnlyList<SearchResult> ranked,
        int finalTopK,
        int mrrK,
        int ndcgK,
        bool usedFallback)
    {
        var relevantSet = question.RelevantChunkIds.ToHashSet();
        var relevanceGrades = question.RelevanceGrades.Count > 0
            ? question.RelevanceGrades
            : question.RelevantChunkIds.ToDictionary(x => x, _ => 1);

        bool hitAtK = ranked.Take(finalTopK).Any(r => relevantSet.Contains(r.ChunkId));
        int rankFirstRelevant = 0;
        double mrrContribution = 0;

        for (int i = 0; i < ranked.Count; i++)
        {
            if (relevantSet.Contains(ranked[i].ChunkId))
            {
                rankFirstRelevant = i + 1;
                mrrContribution = rankFirstRelevant <= mrrK ? 1.0 / rankFirstRelevant : 0;
                break;
            }
        }

        double ndcg = ComputeNdcg(ranked, relevanceGrades, ndcgK);

        var topK = ranked.Take(finalTopK).ToList();
        return new PerQuestionResult(
            experimentId,
            question.QuestionId,
            question.GroupId,
            question.VariantId,
            hitAtK,
            rankFirstRelevant,
            mrrContribution,
            ndcg,
            topK.Select(r => r.ChunkId).ToList(),
            topK.Select(r => r.Score).ToList(),
            usedFallback
        );
    }

    private static double ComputeNdcg(IReadOnlyList<SearchResult> ranked, IReadOnlyDictionary<string, int> relevanceGrades, int k)
    {
        double dcg = 0;
        for (int i = 0; i < Math.Min(k, ranked.Count); i++)
        {
            var chunkId = ranked[i].ChunkId;
            relevanceGrades.TryGetValue(chunkId, out var rel);
            if (rel == 0) continue;
            dcg += (Math.Pow(2, rel) - 1) / Math.Log2(i + 2);
        }

        var sortedRelevant = relevanceGrades.Values.OrderByDescending(v => v).Take(k).ToList();
        double idcg = 0;
        for (int i = 0; i < sortedRelevant.Count; i++)
        {
            var rel = sortedRelevant[i];
            idcg += (Math.Pow(2, rel) - 1) / Math.Log2(i + 2);
        }

        if (idcg == 0) return 0;
        return dcg / idcg;
    }
}
