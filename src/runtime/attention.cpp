#include "swarx/attention.hpp"
#include "swarx/tensor_ops.hpp"

#include <cmath>
#include <stdexcept>

namespace swarx {
namespace {

Tensor linear(const Tensor& x, const Tensor& weight) {
    return matmul(x, weight);
}

Tensor causal_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                        std::size_t heads, float scale) {
    const auto& s = q.shape();
    const std::size_t seq = s[0];
    const std::size_t hidden = s[1];
    const std::size_t head_dim = hidden / heads;
    Tensor output({seq, hidden});

    for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t offset = h * head_dim;
        for (std::size_t i = 0; i < seq; ++i) {
            float max_score = -INFINITY;
            for (std::size_t j = 0; j <= i; ++j) {
                float score = 0.0f;
                for (std::size_t d = 0; d < head_dim; ++d)
                    score += q[i * hidden + offset + d] * k[j * hidden + offset + d];
                score *= scale;
                if (score > max_score) max_score = score;
            }

            float denom = 0.0f;
            for (std::size_t j = 0; j <= i; ++j) {
                float score = 0.0f;
                for (std::size_t d = 0; d < head_dim; ++d)
                    score += q[i * hidden + offset + d] * k[j * hidden + offset + d];
                denom += std::exp(score * scale - max_score);
            }

            for (std::size_t d = 0; d < head_dim; ++d) {
                float value = 0.0f;
                for (std::size_t j = 0; j <= i; ++j) {
                    float score = 0.0f;
                    for (std::size_t kd = 0; kd < head_dim; ++kd)
                        score += q[i * hidden + offset + kd] * k[j * hidden + offset + kd];
                    const float probability = std::exp(score * scale - max_score) / denom;
                    value += probability * v[j * hidden + offset + d];
                }
                output[i * hidden + offset + d] = value;
            }
        }
    }
    return output;
}

} // namespace

CausalSelfAttention::CausalSelfAttention(AttentionConfig config)
    : config_(config),
      query_weight_({config.hidden_size, config.hidden_size}),
      key_weight_({config.hidden_size, config.hidden_size}),
      value_weight_({config.hidden_size, config.hidden_size}),
      output_weight_({config.hidden_size, config.hidden_size}) {
    if (config_.hidden_size == 0 || config_.num_heads == 0 ||
        config_.hidden_size % config_.num_heads != 0) {
        throw std::invalid_argument("invalid attention configuration");
    }

    // Deterministic identity initialization keeps the primitive testable
    // before learned model weights/checkpoint loading are introduced.
    for (std::size_t i = 0; i < config_.hidden_size; ++i) {
        const std::size_t index = i * config_.hidden_size + i;
        query_weight_[index] = 1.0f;
        key_weight_[index] = 1.0f;
        value_weight_[index] = 1.0f;
        output_weight_[index] = 1.0f;
    }
}

Tensor CausalSelfAttention::forward(const Tensor& input) const {
    if (input.shape().size() != 2 || input.shape()[1] != config_.hidden_size)
        throw std::invalid_argument("attention input must be [sequence, hidden_size]");

    const Tensor q = linear(input, query_weight_);
    const Tensor k = linear(input, key_weight_);
    const Tensor v = linear(input, value_weight_);
    const float scale = config_.scale > 0.0f
        ? config_.scale
        : 1.0f / std::sqrt(static_cast<float>(config_.hidden_size / config_.num_heads));
    return linear(causal_attention(q, k, v, config_.num_heads, scale), output_weight_);
}

} // namespace swarx
