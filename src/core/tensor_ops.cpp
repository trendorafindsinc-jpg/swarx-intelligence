#include "swarx/tensor_ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace swarx {
namespace {
void require_same_shape(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("tensor shape mismatch");
    }
}
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("matmul requires compatible 2D tensors");
    }
    const auto m = a.shape()[0];
    const auto k = a.shape()[1];
    const auto n = b.shape()[1];
    Tensor out({m, n});
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (std::size_t x = 0; x < k; ++x) sum += a[i * k + x] * b[x * n + j];
            out[i * n + j] = sum;
        }
    }
    return out;
}

Tensor transpose2d(const Tensor& input) {
    if (input.shape().size() != 2) throw std::invalid_argument("transpose2d requires a 2D tensor");
    const auto rows = input.shape()[0];
    const auto cols = input.shape()[1];
    Tensor out({cols, rows});
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) out[c * rows + r] = input[r * cols + c];
    return out;
}

Tensor softmax(const Tensor& input) {
    if (input.size() == 0) throw std::invalid_argument("softmax requires non-empty tensor");
    Tensor out(input.shape());
    const float max_value = *std::max_element(input.data(), input.data() + input.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < input.size(); ++i) {
        out[i] = std::exp(input[i] - max_value);
        sum += out[i];
    }
    for (std::size_t i = 0; i < out.size(); ++i) out[i] /= sum;
    return out;
}

Tensor rms_norm(const Tensor& input, const Tensor& weight, float epsilon) {
    require_same_shape(input, weight);
    if (input.size() == 0) throw std::invalid_argument("rms_norm requires non-empty tensor");
    float mean_square = 0.0f;
    for (std::size_t i = 0; i < input.size(); ++i) mean_square += input[i] * input[i];
    mean_square /= static_cast<float>(input.size());
    const float scale = 1.0f / std::sqrt(mean_square + epsilon);
    Tensor out(input.shape());
    for (std::size_t i = 0; i < input.size(); ++i) out[i] = input[i] * scale * weight[i];
    return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b);
    Tensor out(a.shape());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    return out;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b);
    Tensor out(a.shape());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
    return out;
}

} // namespace swarx
