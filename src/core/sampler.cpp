#include "swarx/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace swarx {

Sampler::Sampler(std::uint64_t seed) : rng_(seed == 0 ? std::random_device{}() : seed) {}

std::size_t Sampler::greedy(const std::vector<float>& logits) const {
    if (logits.empty()) throw std::invalid_argument("cannot sample an empty logit vector");
    return static_cast<std::size_t>(std::distance(logits.begin(),
        std::max_element(logits.begin(), logits.end())));
}

std::size_t Sampler::sample(const std::vector<float>& logits, float temperature) {
    if (logits.empty()) throw std::invalid_argument("cannot sample an empty logit vector");
    if (!(temperature > 0.0f)) return greedy(logits);

    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> weights;
    weights.reserve(logits.size());
    for (const float logit : logits) {
        weights.push_back(std::exp((static_cast<double>(logit) - max_logit) / temperature));
    }

    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return distribution(rng_);
}

} // namespace swarx
