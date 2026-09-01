#pragma once

#include "swarx/generation.hpp"
#include "swarx/kv_cache.hpp"
#include "swarx/tokenizer.hpp"
#include "swarx/weights.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace swarx {

struct RuntimeStats {
    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t total_tokens = 0;
    double elapsed_ms = 0.0;
    double tokens_per_second = 0.0;
    bool cancelled = false;
    bool stopped_on_eos = false;
    bool hit_context_limit = false;
};

struct InferenceConfig {
    GenerationConfig generation{};
    bool prepend_bos = true;
};

struct InferenceResult {
    std::vector<TokenId> tokens;
    std::string text;
    RuntimeStats stats{};
};

struct BatchInferenceResult {
    std::vector<InferenceResult> results;
    std::size_t total_prompt_tokens = 0;
    std::size_t total_generated_tokens = 0;
    double elapsed_ms = 0.0;
    double generated_tokens_per_second = 0.0;
};

// High-level SWARX-owned inference lifecycle. It keeps model state, tokenizer,
// KV-cache capacity and generation policy behind one stable runtime boundary.
class InferenceSession {
public:
    InferenceSession();
    explicit InferenceSession(TransformerModel model);
    explicit InferenceSession(const std::string& weights_path);

    const TransformerModel& model() const noexcept { return model_; }
    const ModelConfig& model_config() const noexcept { return model_.config(); }
    const KVCache& cache() const noexcept { return cache_; }
    const RuntimeStats& last_stats() const noexcept { return last_stats_; }

    void reset() noexcept;
    void load_weights(const std::string& path);
    void save_weights(const std::string& path) const;

    InferenceResult generate(std::string_view prompt,
                             InferenceConfig config = {},
                             TokenCallback on_token = {},
                             CancelCallback should_cancel = {});

    InferenceResult generate(const std::vector<TokenId>& prompt_tokens,
                             InferenceConfig config = {},
                             TokenCallback on_token = {},
                             CancelCallback should_cancel = {});

    BatchInferenceResult generate_batch(
        const std::vector<std::string>& prompts,
        InferenceConfig config = {});

    BatchInferenceResult generate_batch(
        const std::vector<std::vector<TokenId>>& prompts,
        InferenceConfig config = {});

private:
    void rebuild_runtime();

    TransformerModel model_;
    Tokenizer tokenizer_;
    KVCache cache_;
    std::uint64_t seed_ = 0;
    RuntimeStats last_stats_{};
};

} // namespace swarx
