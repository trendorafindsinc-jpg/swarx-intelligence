#include "swarx/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace swarx {

QuantizedTensor quantize_int8(const Tensor& tensor) {
    QuantizedTensor result;
    result.shape = tensor.shape();
    result.values.resize(tensor.size());

    float max_abs = 0.0f;
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        if (!std::isfinite(tensor[i])) {
            throw std::invalid_argument("SWARX INT8 quantization requires finite values");
        }
        max_abs = std::max(max_abs, std::abs(tensor[i]));
    }

    if (max_abs == 0.0f) {
        result.scale = 1.0f;
        return result;
    }

    result.scale = max_abs / 127.0f;
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        const float scaled = tensor[i] / result.scale;
        const float rounded = std::round(scaled);
        const float clamped = std::clamp(rounded, -127.0f, 127.0f);
        result.values[i] = static_cast<std::int8_t>(clamped);
    }
    return result;
}

Tensor dequantize_int8(const QuantizedTensor& tensor) {
    if (tensor.scale <= 0.0f || !std::isfinite(tensor.scale)) {
        throw std::invalid_argument("SWARX INT8 tensor has an invalid scale");
    }
    Tensor result(tensor.shape);
    if (result.size() != tensor.values.size()) {
        throw std::invalid_argument("SWARX INT8 tensor shape does not match value count");
    }
    for (std::size_t i = 0; i < tensor.values.size(); ++i) {
        result[i] = static_cast<float>(tensor.values[i]) * tensor.scale;
    }
    return result;
}

QuantizationStats quantization_stats(const Tensor& original,
                                     const QuantizedTensor& quantized) {
    if (original.shape() != quantized.shape || original.size() != quantized.size()) {
        throw std::invalid_argument("SWARX quantization statistics shape mismatch");
    }
    QuantizationStats result;
    result.elements = original.size();
    result.bytes_f32 = original.size() * sizeof(float);
    result.bytes_int8 = quantized.bytes();
    result.scale = quantized.scale;
    if (original.size() == 0) return result;

    double total_error = 0.0;
    float maximum_error = 0.0f;
    for (std::size_t i = 0; i < original.size(); ++i) {
        const float restored = static_cast<float>(quantized.values[i]) * quantized.scale;
        const float error = std::abs(original[i] - restored);
        maximum_error = std::max(maximum_error, error);
        total_error += static_cast<double>(error);
    }
    result.max_abs_error = maximum_error;
    result.mean_abs_error = static_cast<float>(total_error / original.size());
    return result;
}

} // namespace swarx
