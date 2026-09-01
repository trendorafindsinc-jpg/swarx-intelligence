#include "swarx/transformer.hpp"
#include "swarx/tensor_ops.hpp"

#include <cmath>
#include <stdexcept>

namespace swarx {
namespace {

Tensor gelu(const Tensor& input) {
    Tensor output(input.shape());
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    constexpr float kCoeff = 0.044715f;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(
            kSqrtTwoOverPi * (x + kCoeff * x * x * x)));
    }
    return output;
}

Tensor rowwise_linear(const Tensor& input, const Tensor& weight) {
    return matmul(input, weight);
}

} // namespace

TransformerBlock::TransformerBlock(TransformerConfig config)
    : config_(config),
      attention_(AttentionConfig{config.hidden_size, config.num_heads, 0.0f}),
      attention_norm_weight_({config.hidden_size}),
      mlp_norm_weight_({config.hidden_size}),
      up_weight_({config.hidden_size, config.intermediate_size}),
      down_weight_({config.intermediate_size, config.hidden_size}) {
    if (config_.hidden_size == 0 || config_.intermediate_size == 0 ||
        config_.num_heads == 0 || config_.hidden_size % config_.num_heads != 0) {
        throw std::invalid_argument("invalid transformer configuration");
    }

    for (std::size_t i = 0; i < config_.hidden_size; ++i) {
        attention_norm_weight_[i] = 1.0f;
        mlp_norm_weight_[i] = 1.0f;
    }

    // Deterministic small weights make the block executable before checkpoint
    // loading is introduced. The diagonal portion preserves signal while the
    // remaining connections are zero until learned weights are loaded.
    const std::size_t diagonal = config_.hidden_size < config_.intermediate_size
        ? config_.hidden_size : config_.intermediate_size;
    for (std::size_t i = 0; i < diagonal; ++i) {
        up_weight_[i * config_.intermediate_size + i] = 1.0f;
        down_weight_[i * config_.hidden_size + i] = 1.0f;
    }
}

Tensor TransformerBlock::forward(const Tensor& input) const {
    if (input.shape().size() != 2 || input.shape()[1] != config_.hidden_size) {
        throw std::invalid_argument("transformer input must be [sequence, hidden_size]");
    }

    // Pre-norm attention + residual.
    Tensor normalized(input.shape());
    for (std::size_t row = 0; row < input.shape()[0]; ++row) {
        Tensor row_input({config_.hidden_size});
        for (std::size_t col = 0; col < config_.hidden_size; ++col)
            row_input[col] = input[row * config_.hidden_size + col];
        const Tensor row_norm = rms_norm(
            row_input, attention_norm_weight_, config_.rms_epsilon);
        for (std::size_t col = 0; col < config_.hidden_size; ++col)
            normalized[row * config_.hidden_size + col] = row_norm[col];
    }

    const Tensor attended = attention_.forward(normalized);
    const Tensor attention_residual = add(input, attended);

    // Pre-norm feed-forward network + residual.
    Tensor mlp_normalized(attention_residual.shape());
    for (std::size_t row = 0; row < attention_residual.shape()[0]; ++row) {
        Tensor row_input({config_.hidden_size});
        for (std::size_t col = 0; col < config_.hidden_size; ++col)
            row_input[col] = attention_residual[row * config_.hidden_size + col];
        const Tensor row_norm = rms_norm(
            row_input, mlp_norm_weight_, config_.rms_epsilon);
        for (std::size_t col = 0; col < config_.hidden_size; ++col)
            mlp_normalized[row * config_.hidden_size + col] = row_norm[col];
    }

    const Tensor hidden = gelu(rowwise_linear(mlp_normalized, up_weight_));
    const Tensor mlp_output = rowwise_linear(hidden, down_weight_);
    return add(attention_residual, mlp_output);
}

} // namespace swarx
