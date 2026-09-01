#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace swarx {

class Sampler {
public:
    explicit Sampler(std::uint64_t seed = 0);

    std::size_t greedy(const std::vector<float>& logits) const;
    std::size_t sample(const std::vector<float>& logits, float temperature);

private:
    std::mt19937_64 rng_;
};

} // namespace swarx
