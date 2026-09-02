#include "swarx/engine.hpp"
#include "swarx/weights.hpp"

#include <sstream>
#include <stdexcept>

namespace swarx {

Engine::Engine(EngineConfig config) : config_(config), sampler_(42) {}

void Engine::load_model(const std::string& weights_path) {
    model_ = std::make_unique<TransformerModel>(WeightsReader::load(weights_path));
}

void Engine::load_tokenizer_vocab(const std::string& vocab_path) {
    tokenizer_.load_vocab(vocab_path);
}

std::vector<TokenId> Engine::tokenize(std::string_view text) const {
    return tokenizer_.encode(text);
}

std::vector<float> Engine::next_token_logits(std::string_view prompt) const {
    if (!model_) throw std::runtime_error("next_token_logits requires a loaded model");

    auto ids = tokenizer_.encode(prompt, /*add_bos=*/true, /*add_eos=*/false);
    if (ids.empty()) ids.push_back(tokenizer_.bos());

    const Tensor logits = model_->next_token_logits(ids);
    return std::vector<float>(logits.data(), logits.data() + logits.size());
}

std::string Engine::generate(std::string_view prompt, std::size_t max_tokens) {
    if (!model_) {
        // Bootstrap path kept for existing smoke tests and pre-model runs.
        std::ostringstream out;
        out << "SWARX Intelligence runtime is ready. Prompt received: " << prompt;
        return out.str();
    }

    const std::size_t limit =
        max_tokens == 0 ? config_.max_new_tokens : max_tokens;

    auto ids = tokenizer_.encode(prompt, /*add_bos=*/true, /*add_eos=*/false);
    if (ids.empty()) ids.push_back(tokenizer_.bos());

    GenerationConfig gen;
    gen.max_new_tokens = limit;
    gen.temperature = config_.temperature;
    gen.eos_token_id = tokenizer_.eos();
    gen.stop_on_eos = true;
    gen.slide_context = true;

    Generator generator(*model_, 42);
    const auto result = generator.generate(std::move(ids), gen);
    return tokenizer_.decode(result.tokens, /*skip_special=*/true);
}

} // namespace swarx