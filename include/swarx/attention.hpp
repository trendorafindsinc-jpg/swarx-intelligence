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

    const Tensor& query_weight() const noexcept { return query_weight_; }
    const Tensor& key_weight() const noexcept { return key_weight_; }
    const Tensor& value_weight() const noexcept { return value_weight_; }
    const Tensor& output_weight() const noexcept { return output_weight_; }

private:
    AttentionConfig config_;
    Tensor query_weight_;
    Tensor key_weight_;
    Tensor value_weight_;
    Tensor output_weight_;
};

} // namespace swarx
