#pragma once

#include "swarx/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace swarx {

// Compact symmetric INT8 tensor representation. A single scale is stored for
// the tensor so the format stays simple and deterministic; per-channel schemes
// can be added later without changing the model-facing API.
struct QuantizedTensor {
    std::vector<std::size_t> shape;
    std::vector<std::int8_t> values;
    float scale = 1.0f;

    std::size_t size() const noexcept { return values.size(); }
    std::size_t bytes() const noexcept { return values.size(); }
};

struct QuantizationStats {
    std::size_t elements = 0;
    std::size_t bytes_f32 = 0;
    std::size_t bytes_int8 = 0;
    float scale = 1.0f;
    float max_abs_error = 0.0f;
    float mean_abs_error = 0.0f;
};

QuantizedTensor quantize_int8(const Tensor& tensor);
Tensor dequantize_int8(const QuantizedTensor& tensor);
QuantizationStats quantization_stats(const Tensor& original,
                                     const QuantizedTensor& quantized);

} // namespace swarx
