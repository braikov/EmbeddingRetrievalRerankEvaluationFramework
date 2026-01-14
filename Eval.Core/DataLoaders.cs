using System.Text.Json;

namespace Eval.Core;

public static class DataLoaders
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        PropertyNameCaseInsensitive = true
    };

    public static EvaluationConfig LoadConfig(string path)
    {
        var json = File.ReadAllText(path);
        var config = JsonSerializer.Deserialize<EvaluationConfig>(json, JsonOptions);
        if (config == null)
        {
            throw new InvalidOperationException("Unable to parse config.json");
        }

        return config;
    }

    public static Dictionary<string, CorpusChunk> LoadCorpus(string path)
    {
        var dict = new Dictionary<string, CorpusChunk>();
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var chunk = JsonSerializer.Deserialize<CorpusChunk>(line, JsonOptions);
            if (chunk == null) continue;
            dict[chunk.ChunkId] = chunk;
        }
        return dict;
    }

    public static List<Question> LoadQuestions(string path)
    {
        var result = new List<Question>();
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var dto = JsonSerializer.Deserialize<QuestionDto>(line, JsonOptions);
            if (dto == null) continue;
            var relevanceGrades = dto.RelevanceGrades ?? new Dictionary<string, int>();
            result.Add(new Question(
                dto.QuestionId,
                dto.GroupId,
                dto.VariantId,
                dto.Question,
                dto.RelevantChunkIds ?? new List<string>(),
                relevanceGrades
            ));
        }
        return result;
    }

    private class QuestionDto
    {
        public string QuestionId { get; set; } = string.Empty;
        public string GroupId { get; set; } = string.Empty;
        public string VariantId { get; set; } = string.Empty;
        public string Question { get; set; } = string.Empty;
        public List<string>? RelevantChunkIds { get; set; }
        public Dictionary<string, int>? RelevanceGrades { get; set; }
    }
}
