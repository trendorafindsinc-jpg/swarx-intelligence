#include "swarx/backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace swarx {

BackendCapabilities CpuBackend::capabilities() const noexcept {
    BackendCapabilities result;
    result.cpu = true;
    result.int8 = true;
    return result;
}

bool CpuBackend::supports(BackendOp op) const noexcept {
    switch (op) {
    case BackendOp::MatMul:
    case BackendOp::RMSNorm:
    case BackendOp::Softmax:
    case BackendOp::GELU:
        return true;
    }
    return false;
}

Tensor CpuBackend::matmul(const Tensor& a, const Tensor& b) const {
    if (a.shape().size() != 2 || b.shape().size() != 2 ||
        a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("SWARX CPU matmul requires compatible 2D tensors");
    }

    const std::size_t m = a.shape()[0];
    const std::size_t k = a.shape()[1];
    const std::size_t n = b.shape()[1];
    Tensor result({m, n});

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            const float av = a[i * k + p];
            for (std::size_t j = 0; j < n; ++j) {
                result[i * n + j] += av * b[p * n + j];
            }
        }
    }
    return result;
}

Tensor CpuBackend::rms_norm(const Tensor& input, const Tensor& weight,
                            float epsilon) const {
    if (input.shape().empty() || weight.shape().size() != 1 ||
        input.shape().back() != weight.shape()[0] || epsilon <= 0.0f) {
        throw std::invalid_argument("SWARX CPU RMSNorm shape or epsilon mismatch");
    }

    const std::size_t width = input.shape().back();
    const std::size_t rows = input.size() / width;
    Tensor result(input.shape());
    for (std::size_t row = 0; row < rows; ++row) {
        float mean_square = 0.0f;
        for (std::size_t i = 0; i < width; ++i) {
            const float value = input[row * width + i];
            mean_square += value * value;
        }
        mean_square /= static_cast<float>(width);
        const float inv_rms = 1.0f / std::sqrt(mean_square + epsilon);
        for (std::size_t i = 0; i < width; ++i) {
            result[row * width + i] = input[row * width + i] * inv_rms * weight[i];
        }
    }
    return result;
}

Tensor CpuBackend::softmax(const Tensor& input) const {
    if (input.shape().empty()) {
        throw std::invalid_argument("SWARX CPU softmax requires a non-empty tensor");
    }
    const std::size_t width = input.shape().back();
    const std::size_t rows = input.size() / width;
    Tensor result(input.shape());

    for (std::size_t row = 0; row < rows; ++row) {
        const auto begin = input.data() + row * width;
        const float maximum = *std::max_element(begin, begin + width);
        float sum = 0.0f;
        for (std::size_t i = 0; i < width; ++i) {
            result[row * width + i] = std::exp(begin[i] - maximum);
            sum += result[row * width + i];
        }
        if (!(sum > 0.0f) || !std::isfinite(sum)) {
            throw std::runtime_error("SWARX CPU softmax produced an invalid sum");
        }
        for (std::size_t i = 0; i < width; ++i) {
            result[row * width + i] /= sum;
        }
    }
    return result;
}

Tensor CpuBackend::gelu(const Tensor& input) const {
    Tensor result(input.shape());
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    constexpr float kCubic = 0.044715f;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        result[i] = 0.5f * x * (1.0f + std::tanh(kSqrtTwoOverPi *
                         (x + kCubic * x * x * x)));
    }
    return result;
}

const ComputeBackend& BackendRegistry::cpu() noexcept {
    static const CpuBackend backend;
    return backend;
}

const ComputeBackend& BackendRegistry::default_backend() noexcept {
    return cpu();
}

} // namespace swarx
