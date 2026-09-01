#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace swarx {

struct SamplingConfig {
    float temperature = 0.0f;
    std::size_t top_k = 0;       // 0 = disabled
    float top_p = 1.0f;          // 1 = disabled
};

class Sampler {
public:
    explicit Sampler(std::uint64_t seed = 0);

    std::size_t greedy(const std::vector<float>& logits) const;
    std::size_t sample(const std::vector<float>& logits, float temperature);
    std::size_t sample(const std::vector<float>& logits, const SamplingConfig& config);

private:
    std::mt19937_64 rng_;
};

} // namespace swarx
