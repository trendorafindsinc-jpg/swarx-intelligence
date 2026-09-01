#pragma once

#include "swarx/tensor.hpp"
#include "swarx/transformer.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace swarx {

class WeightsReader;

// Canonical SWARX-0.1 foundation dimensions. The weights reader still accepts
// dimensions serialized in the artifact, so these defaults only define the
// model constructed without a checkpoint.
struct ModelConfig {
    std::size_t vocab_size = 8192;
    std::size_t hidden_size = 512;
    std::size_t intermediate_size = 2048;
    std::size_t num_heads = 8;
    std::size_t num_layers = 8;
    std::size_t max_sequence_length = 2048;
    float rms_epsilon = 1e-5f;
};

class TransformerModel {
public:
    explicit TransformerModel(ModelConfig config = {});

    const ModelConfig& config() const noexcept { return config_; }
    Tensor forward(const std::vector<std::uint32_t>& tokens) const;
    Tensor logits(const Tensor& hidden_state) const;
    Tensor next_token_logits(const std::vector<std::uint32_t>& tokens) const;

    const Tensor& token_embedding() const noexcept { return token_embedding_; }
    const Tensor& position_embedding() const noexcept { return position_embedding_; }
    const Tensor& final_norm_weight() const noexcept { return final_norm_weight_; }
    const Tensor& lm_head() const noexcept { return lm_head_; }
    const std::vector<TransformerBlock>& blocks() const noexcept { return blocks_; }

private:
    friend class WeightsReader;
    ModelConfig config_;
    Tensor token_embedding_;
    Tensor position_embedding_;
    Tensor final_norm_weight_;
    Tensor lm_head_;
    std::vector<TransformerBlock> blocks_;
};

} // namespace swarx
