#pragma once

#include "swarx/tensor.hpp"

#include <cstddef>

namespace swarx {

struct AttentionConfig {
    std::size_t hidden_size = 64;
    std::size_t num_heads = 4;
    float scale = 0.0f; // 0 = derive from head dimension.
};

class CausalSelfAttention {
public:
    explicit CausalSelfAttention(AttentionConfig config = {});

    // Input/output shape: [sequence, hidden_size].
    // The mask prevents every position from attending to future tokens.
    Tensor forward(const Tensor& input) const;

private:
    AttentionConfig config_;
    Tensor query_weight_;
    Tensor key_weight_;
    Tensor value_weight_;
    Tensor output_weight_;
};

} // namespace swarx
