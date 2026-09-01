#include "swarx/engine.hpp"
#include "swarx/model.hpp"
#include "swarx/sampler.hpp"
#include "swarx/tensor.hpp"
#include "swarx/transformer.hpp"

#include <cassert>
#include <cmath>
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

    return 0;
}
