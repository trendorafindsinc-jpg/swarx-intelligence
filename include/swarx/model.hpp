#pragma once

#include "swarx/tensor.hpp"
#include "swarx/transformer.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace swarx {

struct ModelConfig {
    std::size_t vocab_size = 512;
    std::size_t hidden_size = 64;
    std::size_t intermediate_size = 256;
    std::size_t num_heads = 4;
    std::size_t num_layers = 4;
    std::size_t max_sequence_length = 2048;
    float rms_epsilon = 1e-5f;
};

class TransformerModel {
public:
    explicit TransformerModel(ModelConfig config = {});

    const ModelConfig& config() const noexcept { return config_; }

    // Converts token ids into final hidden states. Shape: [sequence, hidden].
    Tensor forward(const std::vector<std::uint32_t>& tokens) const;

    // Produces next-token logits. Shape: [vocab_size].
    Tensor logits(const Tensor& hidden_state) const;

    // Full forward pass for one next-token prediction.
    Tensor next_token_logits(const std::vector<std::uint32_t>& tokens) const;

private:
    ModelConfig config_;
    Tensor token_embedding_;
    Tensor position_embedding_;
    Tensor final_norm_weight_;
    Tensor lm_head_;
    std::vector<TransformerBlock> blocks_;
};

} // namespace swarx
