#pragma once

#include "swarx/attention.hpp"
#include "swarx/tensor.hpp"

#include <cstddef>

namespace swarx {

class WeightsReader;

struct TransformerConfig {
    std::size_t hidden_size = 64;
    std::size_t intermediate_size = 256;
    std::size_t num_heads = 4;
    float rms_epsilon = 1e-5f;
};

class TransformerBlock {
public:
    explicit TransformerBlock(TransformerConfig config = {});
    Tensor forward(const Tensor& input) const;

    const CausalSelfAttention& attention() const noexcept { return attention_; }
    const Tensor& attention_norm_weight() const noexcept { return attention_norm_weight_; }
    const Tensor& mlp_norm_weight() const noexcept { return mlp_norm_weight_; }
    const Tensor& up_weight() const noexcept { return up_weight_; }
    const Tensor& down_weight() const noexcept { return down_weight_; }

private:
    friend class WeightsReader;
    TransformerConfig config_;
    CausalSelfAttention attention_;
    Tensor attention_norm_weight_;
    Tensor mlp_norm_weight_;
    Tensor up_weight_;
    Tensor down_weight_;
};

} // namespace swarx
