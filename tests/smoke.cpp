#include "swarx/engine.hpp"
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

    swarx::TransformerConfig config;
    config.hidden_size = 8;
    config.intermediate_size = 16;
    config.num_heads = 2;
    swarx::TransformerBlock block(config);

    swarx::Tensor sequence({3, 8});
    for (std::size_t i = 0; i < sequence.size(); ++i)
        sequence[i] = static_cast<float>(i + 1) * 0.05f;

    const swarx::Tensor output = block.forward(sequence);
    assert(output.shape().size() == 2);
    assert(output.shape()[0] == 3);
    assert(output.shape()[1] == 8);
    for (std::size_t i = 0; i < output.size(); ++i)
        assert(std::isfinite(output[i]));

    return 0;
}
