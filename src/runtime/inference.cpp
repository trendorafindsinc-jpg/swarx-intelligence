#include "swarx/inference.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace swarx {

InferenceSession::InferenceSession(TransformerModel model)
    : model_(std::move(model)) {
    rebuild_runtime();
}

InferenceSession::InferenceSession(const std::string& weights_path)
    : model_(WeightsReader::load(weights_path)) {
    rebuild_runtime();
}

void InferenceSession::rebuild_runtime() {
    const auto& cfg = model_.config();
    if (cfg.vocab_size == 0 || cfg.hidden_size == 0 || cfg.num_layers == 0 ||
        cfg.max_sequence_length == 0) {
        throw std::invalid_argument("invalid SWARX model configuration");
    }
    if (cfg.num_heads == 0 || cfg.hidden_size % cfg.num_heads != 0) {
        throw std::invalid_argument("hidden size must be divisible by num heads");
    }
    cache_ = KVCache(cfg.num_layers, cfg.max_sequence_length, cfg.hidden_size);
}

void InferenceSession::reset() noexcept {
    cache_.reset();
    last_stats_ = {};
}

void InferenceSession::load_weights(const std::string& path) {
    model_ = WeightsReader::load(path);
    rebuild_runtime();
    last_stats_ = {};
}

void InferenceSession::save_weights(const std::string& path) const {
    WeightsWriter::write(path, model_);
}

InferenceResult InferenceSession::generate(std::string_view prompt,
                                           InferenceConfig config,
                                           TokenCallback on_token,
                                           CancelCallback should_cancel) {
    auto tokens = tokenizer_.encode(prompt);
    return generate(tokens, config, std::move(on_token), std::move(should_cancel));
}

InferenceResult InferenceSession::generate(const std::vector<TokenId>& prompt_tokens,
                                           InferenceConfig config,
                                           TokenCallback on_token,
                                           CancelCallback should_cancel) {
    reset();

    std::vector<TokenId> prompt(prompt_tokens.begin(), prompt_tokens.end());
    if (config.prepend_bos && (prompt.empty() || prompt.front() != tokenizer_.bos())) {
        prompt.insert(prompt.begin(), tokenizer_.bos());
    }

    const auto start = std::chrono::steady_clock::now();
    Generator generator(model_, seed_);
    auto result = generator.generate(std::move(prompt), config.generation,
                                     std::move(on_token), std::move(should_cancel));
    const auto end = std::chrono::steady_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    last_stats_.generated_tokens = result.generated_tokens;
    last_stats_.total_tokens = result.tokens.size();
    last_stats_.prompt_tokens = result.tokens.size() >= result.generated_tokens
        ? result.tokens.size() - result.generated_tokens : 0;
    last_stats_.elapsed_ms = elapsed_ms;
    last_stats_.tokens_per_second =
        elapsed_ms > 0.0 ? (1000.0 * result.generated_tokens / elapsed_ms) : 0.0;
    last_stats_.cancelled = result.cancelled;
    last_stats_.stopped_on_eos = result.stopped_on_eos;
    last_stats_.hit_context_limit = result.hit_context_limit;

    InferenceResult output;
    output.tokens = std::move(result.tokens);
    output.text = tokenizer_.decode(output.tokens);
    output.stats = last_stats_;
    return output;
}

} // namespace swarx
