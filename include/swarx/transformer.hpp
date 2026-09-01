#pragma once

#include "swarx/attention.hpp"
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
    CausalSelfAttention attention_;
    Tensor attention_norm_weight_;
    Tensor mlp_norm_weight_;
    Tensor up_weight_;
    Tensor down_weight_;
};

} // namespace swarx
