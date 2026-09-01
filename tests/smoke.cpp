#include "swarx/engine.hpp"
#include "swarx/generation.hpp"
#include "swarx/model.hpp"
#include "swarx/sampler.hpp"
#include "swarx/tensor.hpp"
#include "swarx/transformer.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

int main() {
    swarx::Tensor tensor({2, 3});
    assert(tensor.size() == 6);

    swarx::Sampler sampler(1234);
    assert(sampler.greedy({-1.0f, 3.0f, 2.0f}) == 1);

    swarx::Engine engine;
    const auto tokens = engine.tokenize("hello world");
    assert(tokens.size() == 2);
    assert(engine.generate("hello").find("SWARX Intelligence runtime is ready") != std::string::npos);

    swarx::TransformerConfig block_config;
    block_config.hidden_size = 8;
    block_config.intermediate_size = 16;
    block_config.num_heads = 2;
    swarx::TransformerBlock block(block_config);

    swarx::Tensor sequence({3, 8});
    for (std::size_t i = 0; i < sequence.size(); ++i)
        sequence[i] = static_cast<float>(i + 1) * 0.05f;

    const swarx::Tensor output = block.forward(sequence);
    assert(output.shape().size() == 2);
    assert(output.shape()[0] == 3);
    assert(output.shape()[1] == 8);
    for (std::size_t i = 0; i < output.size(); ++i)
        assert(std::isfinite(output[i]));

    swarx::ModelConfig model_config;
    model_config.vocab_size = 64;
    model_config.hidden_size = 16;
    model_config.intermediate_size = 32;
    model_config.num_heads = 4;
    model_config.num_layers = 3;
    model_config.max_sequence_length = 32;

    swarx::TransformerModel model(model_config);
    const std::vector<std::uint32_t> prompt_tokens = {1, 7, 12, 4};
    const swarx::Tensor hidden = model.forward(prompt_tokens);
    assert(hidden.shape().size() == 2);
    assert(hidden.shape()[0] == prompt_tokens.size());
    assert(hidden.shape()[1] == model_config.hidden_size);
    for (std::size_t i = 0; i < hidden.size(); ++i)
        assert(std::isfinite(hidden[i]));

    const swarx::Tensor next_logits = model.next_token_logits(prompt_tokens);
    assert(next_logits.shape().size() == 1);
    assert(next_logits.shape()[0] == model_config.vocab_size);
    for (std::size_t i = 0; i < next_logits.size(); ++i)
        assert(std::isfinite(next_logits[i]));

    const std::size_t greedy_token = sampler.greedy(
        std::vector<float>(next_logits.data(), next_logits.data() + next_logits.size()));
    assert(greedy_token < model_config.vocab_size);

    // Autoregressive generation: the runtime must produce real token ids and
    // invoke the streaming callback once per generated token.
    swarx::GenerationConfig generation_config;
    generation_config.max_new_tokens = 6;
    generation_config.temperature = 0.0f;
    generation_config.repetition_penalty = 1.1f;
    generation_config.slide_context = true;

    swarx::Generator generator(model, 1234);
    std::size_t streamed = 0;
    const auto generated = generator.generate(
        prompt_tokens, generation_config,
        [&](std::uint32_t token) {
            ++streamed;
            assert(token < model_config.vocab_size);
        });

    assert(generated.tokens.size() == prompt_tokens.size() + 6);
    assert(generated.generated_tokens == 6);
    assert(streamed == 6);
    assert(!generated.stopped_on_eos);

    // Context sliding keeps generation valid after the model context fills.
    swarx::GenerationConfig sliding;
    sliding.max_new_tokens = 5;
    sliding.temperature = 0.0f;
    sliding.slide_context = true;
    const auto long_generated = generator.generate(
        std::vector<std::uint32_t>(model_config.max_sequence_length, 3), sliding);
    assert(long_generated.generated_tokens == 5);
    assert(!long_generated.hit_context_limit);

    return 0;
}
