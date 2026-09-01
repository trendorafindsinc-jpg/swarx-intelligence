#include "swarx/transformer.hpp"
#include "swarx/tensor_ops.hpp"

#include <stdexcept>

namespace swarx {

TransformerBlock::TransformerBlock(TransformerConfig config)
    : config_(config), norm_weight_({config.hidden_size}) {
    if (config_.hidden_size == 0 || config_.num_heads == 0 ||
        config_.hidden_size % config_.num_heads != 0) {
        throw std::invalid_argument("invalid transformer configuration");
    }
    for (std::size_t i = 0; i < norm_weight_.size(); ++i) norm_weight_[i] = 1.0f;
}

Tensor TransformerBlock::forward(const Tensor& input) const {
    if (input.shape().size() != 2 || input.shape()[1] != config_.hidden_size) {
        throw std::invalid_argument("transformer input must be [sequence, hidden_size]");
    }

    // First executable Transformer primitive: pre-normalization plus residual.
    // Attention/MLP weights will be introduced behind this stable API next.
    Tensor output(input.shape());
    for (std::size_t row = 0; row < input.shape()[0]; ++row) {
        Tensor row_input({config_.hidden_size});
        for (std::size_t col = 0; col < config_.hidden_size; ++col)
            row_input[col] = input[row * config_.hidden_size + col];
        const Tensor normalized = rms_norm(row_input, norm_weight_, config_.rms_epsilon);
        for (std::size_t col = 0; col < config_.hidden_size; ++col)
            output[row * config_.hidden_size + col] = input[row * config_.hidden_size + col] + normalized[col];
    }
    return output;
}

} // namespace swarx
