#pragma once

#include "swarx/generation.hpp"
#include "swarx/model.hpp"
#include "swarx/sampler.hpp"
#include "swarx/tokenizer.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace swarx {

struct EngineConfig {
    std::size_t context_size = 2048;
    float temperature = 0.0f; // greedy by default for deterministic nano tests
    std::size_t max_new_tokens = 64;
};

class Engine {
public:
    explicit Engine(EngineConfig config = {});

    const EngineConfig& config() const noexcept { return config_; }
    bool has_model() const noexcept { return model_ != nullptr; }

    // Load exported SWARX v2 weights (model.swarx).
    void load_model(const std::string& weights_path);

    // Optional SentencePiece .vocab so encode/decode match training.
    void load_tokenizer_vocab(const std::string& vocab_path);

    Tokenizer& tokenizer() noexcept { return tokenizer_; }
    const Tokenizer& tokenizer() const noexcept { return tokenizer_; }

    std::vector<TokenId> tokenize(std::string_view text) const;

    // Return the raw next-token logits for a prompt. Intended for
    // deterministic Python/C++ numerical compatibility validation.
    std::vector<float> next_token_logits(std::string_view prompt) const;

    // If a model is loaded: real Transformer generation.
    // Otherwise: bootstrap readiness string (preserves existing smoke behavior).
    std::string generate(std::string_view prompt, std::size_t max_tokens = 0);

private:
    EngineConfig config_;
    Tokenizer tokenizer_;
    Sampler sampler_;
    std::unique_ptr<TransformerModel> model_;
};

} // namespace swarx