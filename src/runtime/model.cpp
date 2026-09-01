#include "swarx/model.hpp"
#include "swarx/tensor_ops.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace swarx {
namespace {

float init_value(std::size_t i, std::size_t j, float scale) {
    const std::uint32_t x = static_cast<std::uint32_t>((i + 1) * 2654435761u ^ (j + 17) * 2246822519u);
    const float unit = static_cast<float>(x % 2001u) / 1000.0f - 1.0f;
    return unit * scale;
}

Tensor make_embedding(std::size_t rows, std::size_t cols, float scale) {
    Tensor out({rows, cols});
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            out[r * cols + c] = init_value(r, c, scale);
    return out;
}

Tensor make_vector(std::size_t size, float value) {
    Tensor out({size});
    for (std::size_t i = 0; i < size; ++i) out[i] = value;
    return out;
}

Tensor rms_norm_rows(const Tensor& input, const Tensor& weight, float epsilon) {
    if (input.shape().size() != 2 || weight.shape().size() != 1 ||
        input.shape()[1] != weight.shape()[0]) {
        throw std::invalid_argument("rms_norm_rows requires [sequence, hidden] input and [hidden] weight");
    }

    const std::size_t sequence = input.shape()[0];
    const std::size_t hidden = input.shape()[1];
    Tensor output(input.shape());

    for (std::size_t row = 0; row < sequence; ++row) {
        float mean_square = 0.0f;
        const std::size_t offset = row * hidden;
        for (std::size_t col = 0; col < hidden; ++col) {
            const float value = input[offset + col];
            mean_square += value * value;
        }
        mean_square /= static_cast<float>(hidden);
        const float scale = 1.0f / std::sqrt(mean_square + epsilon);
        for (std::size_t col = 0; col < hidden; ++col)
            output[offset + col] = input[offset + col] * scale * weight[col];
    }
    return output;
}

} // namespace

TransformerModel::TransformerModel(ModelConfig config)
    : config_(config),
      token_embedding_(make_embedding(config.vocab_size, config.hidden_size,
                                      1.0f / std::sqrt(static_cast<float>(config.hidden_size)))),
      position_embedding_(make_embedding(config.max_sequence_length, config.hidden_size,
                                         1.0f / std::sqrt(static_cast<float>(config.hidden_size)))),
      final_norm_weight_(make_vector(config.hidden_size, 1.0f)),
      lm_head_(make_embedding(config.vocab_size, config.hidden_size,
                              1.0f / std::sqrt(static_cast<float>(config.hidden_size)))) {
    if (config_.hidden_size == 0 || config_.vocab_size == 0 || config_.num_layers == 0)
        throw std::invalid_argument("SWARX model dimensions must be non-zero");
    if (config_.max_sequence_length == 0)
        throw std::invalid_argument("SWARX model context length must be non-zero");
    if (config_.num_heads == 0 || config_.hidden_size % config_.num_heads != 0)
        throw std::invalid_argument("hidden_size must be divisible by num_heads");
    blocks_.reserve(config_.num_layers);
    TransformerConfig block_config;
    block_config.hidden_size = config_.hidden_size;
    block_config.intermediate_size = config_.intermediate_size;
    block_config.num_heads = config_.num_heads;
    block_config.rms_epsilon = config_.rms_epsilon;
    for (std::size_t i = 0; i < config_.num_layers; ++i)
        blocks_.emplace_back(block_config);
}

Tensor TransformerModel::forward(const std::vector<std::uint32_t>& tokens) const {
    if (tokens.empty()) return Tensor({0, config_.hidden_size});
    if (tokens.size() > config_.max_sequence_length)
        throw std::invalid_argument("token sequence exceeds SWARX context length");

    Tensor hidden({tokens.size(), config_.hidden_size});
    for (std::size_t pos = 0; pos < tokens.size(); ++pos) {
        const std::size_t token = static_cast<std::size_t>(tokens[pos]) % config_.vocab_size;
        for (std::size_t h = 0; h < config_.hidden_size; ++h) {
            hidden[pos * config_.hidden_size + h] =
                token_embedding_[token * config_.hidden_size + h] +
                position_embedding_[pos * config_.hidden_size + h];
        }
    }

    for (const auto& block : blocks_) hidden = block.forward(hidden);
    return rms_norm_rows(hidden, final_norm_weight_, config_.rms_epsilon);
}

Tensor TransformerModel::logits(const Tensor& hidden_state) const {
    if (hidden_state.shape().size() != 1 || hidden_state.shape()[0] != config_.hidden_size)
        throw std::invalid_argument("logits expects one hidden vector");
    Tensor out({config_.vocab_size});
    for (std::size_t v = 0; v < config_.vocab_size; ++v) {
        float sum = 0.0f;
        for (std::size_t h = 0; h < config_.hidden_size; ++h)
            sum += hidden_state[h] * lm_head_[v * config_.hidden_size + h];
        out[v] = sum;
    }
    return out;
}

Tensor TransformerModel::next_token_logits(const std::vector<std::uint32_t>& tokens) const {
    if (tokens.empty()) throw std::invalid_argument("next_token_logits requires at least one token");
    const Tensor hidden = forward(tokens);
    Tensor last({config_.hidden_size});
    const std::size_t offset = (tokens.size() - 1) * config_.hidden_size;
    for (std::size_t h = 0; h < config_.hidden_size; ++h) last[h] = hidden[offset + h];
    return logits(last);
}

} // namespace swarx
