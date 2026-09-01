#pragma once

#include "swarx/tensor.hpp"

#include <cstddef>

namespace swarx {

struct TransformerConfig {
    std::size_t hidden_size = 64;
    std::size_t intermediate_size = 256;
    std::size_t num_heads = 4;
    float rms_epsilon = 1e-5f;
};

class TransformerBlock {
public:
    explicit TransformerBlock(TransformerConfig config = {});

    // Runs one decoder-style transformer block over [sequence, hidden].
    Tensor forward(const Tensor& input) const;

private:
    TransformerConfig config_;
    Tensor norm_weight_;
};

} // namespace swarx
