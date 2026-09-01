#include "swarx/backend.hpp"
#include "swarx/quantization.hpp"
#include "swarx/tensor.hpp"

#include <cassert>
#include <cmath>

int main() {
    const auto& backend = swarx::BackendRegistry::default_backend();
    assert(backend.name() == "cpu");
    assert(backend.capabilities().cpu);
    assert(backend.capabilities().int8);
    assert(backend.supports(swarx::BackendOp::MatMul));
    assert(backend.supports(swarx::BackendOp::RMSNorm));
    assert(backend.supports(swarx::BackendOp::Softmax));
    assert(backend.supports(swarx::BackendOp::GELU));

    swarx::Tensor a({2, 3});
    swarx::Tensor b({3, 2});
    a[0] = 1.0f; a[1] = 2.0f; a[2] = 3.0f;
    a[3] = 4.0f; a[4] = 5.0f; a[5] = 6.0f;
    b[0] = 1.0f; b[1] = 2.0f;
    b[2] = 3.0f; b[3] = 4.0f;
    b[4] = 5.0f; b[5] = 6.0f;
    const auto product = backend.matmul(a, b);
    assert(product.shape() == std::vector<std::size_t>({2, 2}));
    assert(std::fabs(product[0] - 22.0f) < 1e-6f);
    assert(std::fabs(product[1] - 28.0f) < 1e-6f);
    assert(std::fabs(product[2] - 49.0f) < 1e-6f);
    assert(std::fabs(product[3] - 64.0f) < 1e-6f);

    swarx::Tensor norm_input({2, 4});
    for (std::size_t i = 0; i < norm_input.size(); ++i)
        norm_input[i] = static_cast<float>(i + 1);
    swarx::Tensor norm_weight({4});
    for (std::size_t i = 0; i < norm_weight.size(); ++i) norm_weight[i] = 1.0f;
    const auto normalized = backend.rms_norm(norm_input, norm_weight, 1e-5f);
    assert(normalized.shape() == norm_input.shape());
    for (std::size_t row = 0; row < 2; ++row) {
        float sum_sq = 0.0f;
        for (std::size_t i = 0; i < 4; ++i)
            sum_sq += normalized[row * 4 + i] * normalized[row * 4 + i];
        assert(std::fabs(sum_sq / 4.0f - 1.0f) < 1e-3f);
    }

    swarx::Tensor logits({2, 3});
    logits[0] = 1.0f; logits[1] = 2.0f; logits[2] = 3.0f;
    logits[3] = 1000.0f; logits[4] = 1000.0f; logits[5] = 1000.0f;
    const auto probabilities = backend.softmax(logits);
    for (std::size_t row = 0; row < 2; ++row) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < 3; ++i) {
            const float value = probabilities[row * 3 + i];
            assert(value >= 0.0f && value <= 1.0f);
            assert(std::isfinite(value));
            sum += value;
        }
        assert(std::fabs(sum - 1.0f) < 1e-6f);
    }

    const auto activated = backend.gelu(logits);
    assert(activated.shape() == logits.shape());
    for (std::size_t i = 0; i < activated.size(); ++i)
        assert(std::isfinite(activated[i]));

    swarx::Tensor weights({8});
    const float values[] = {-3.0f, -1.5f, -0.5f, 0.0f, 0.25f, 1.0f, 2.0f, 3.0f};
    for (std::size_t i = 0; i < 8; ++i) weights[i] = values[i];
    const auto quantized = swarx::quantize_int8(weights);
    assert(quantized.shape == weights.shape());
    assert(quantized.size() == weights.size());
    assert(quantized.scale > 0.0f);
    assert(quantized.bytes() == weights.size());

    const auto restored = swarx::dequantize_int8(quantized);
    const auto stats = swarx::quantization_stats(weights, quantized);
    assert(restored.shape() == weights.shape());
    assert(stats.elements == weights.size());
    assert(stats.bytes_f32 == weights.size() * sizeof(float));
    assert(stats.bytes_int8 == weights.size());
    assert(stats.max_abs_error < 0.02f);
    assert(stats.mean_abs_error < 0.01f);

    swarx::Tensor zeros({4});
    const auto zero_quantized = swarx::quantize_int8(zeros);
    assert(zero_quantized.scale == 1.0f);
    const auto zero_restored = swarx::dequantize_int8(zero_quantized);
    for (std::size_t i = 0; i < zero_restored.size(); ++i)
        assert(zero_restored[i] == 0.0f);

    return 0;
}
