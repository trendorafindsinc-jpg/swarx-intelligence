#include "swarx/generation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace swarx {
namespace {

void apply_repetition_penalty(std::vector<float>& logits,
                              const std::vector<std::uint32_t>& context,
                              float penalty) {
    if (penalty <= 1.0f) return;
    for (const auto token : context) {
        const std::size_t id = static_cast<std::size_t>(token);
        if (id >= logits.size()) continue;
        if (logits[id] >= 0.0f) logits[id] /= penalty;
        else logits[id] *= penalty;
    }
}

} // namespace

Generator::Generator(const TransformerModel& model, std::uint64_t seed)
    : model_(model), seed_(seed) {}

GenerationResult Generator::generate(std::vector<std::uint32_t> prompt,
                                     GenerationConfig config,
                                     TokenCallback on_token,
                                     CancelCallback should_cancel) const {
    const auto& model_config = model_.config();
    if (prompt.empty()) throw std::invalid_argument("SWARX generation requires a non-empty prompt");
    if (config.max_new_tokens == 0) return {std::move(prompt), 0, false, false, false};
    if (config.repetition_penalty < 1.0f)
        throw std::invalid_argument("repetition_penalty must be >= 1.0");
    if (config.temperature < 0.0f)
        throw std::invalid_argument("temperature must be >= 0");
    if (!(config.top_p > 0.0f && config.top_p <= 1.0f))
        throw std::invalid_argument("top_p must be in (0, 1]");

    if (prompt.size() > model_config.max_sequence_length) {
        if (!config.slide_context) throw std::invalid_argument("prompt exceeds SWARX context length");
        const auto first = prompt.end() - static_cast<std::ptrdiff_t>(model_config.max_sequence_length);
        prompt.erase(prompt.begin(), first);
    }

    Sampler sampler(seed_);
    GenerationResult result;
    result.tokens = std::move(prompt);

    for (std::size_t step = 0; step < config.max_new_tokens; ++step) {
        if (should_cancel && should_cancel()) { result.cancelled = true; break; }

        std::vector<std::uint32_t> context = result.tokens;
        if (context.size() > model_config.max_sequence_length) {
            if (!config.slide_context) { result.hit_context_limit = true; break; }
            const auto first = context.end() - static_cast<std::ptrdiff_t>(model_config.max_sequence_length);
            context.erase(context.begin(), first);
        }

        Tensor logits_tensor = model_.next_token_logits(context);
        std::vector<float> logits(logits_tensor.data(), logits_tensor.data() + logits_tensor.size());
        apply_repetition_penalty(logits, context, config.repetition_penalty);

        const std::size_t next = sampler.sample(logits, SamplingConfig{
            config.temperature, config.top_k, config.top_p});
        const auto token = static_cast<std::uint32_t>(next);
        result.tokens.push_back(token);
        ++result.generated_tokens;
        if (on_token) on_token(token);

        if (config.stop_on_eos && token == config.eos_token_id) {
            result.stopped_on_eos = true;
            break;
        }
        if (result.tokens.size() >= model_config.max_sequence_length &&
            step + 1 < config.max_new_tokens && !config.slide_context) {
            result.hit_context_limit = true;
            break;
        }
    }
    return result;
}

} // namespace swarx
