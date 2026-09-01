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
    return sample(logits, SamplingConfig{temperature, 0, 1.0f});
}

std::size_t Sampler::sample(const std::vector<float>& logits, const SamplingConfig& config) {
    if (logits.empty()) throw std::invalid_argument("cannot sample an empty logit vector");
    if (config.temperature <= 0.0f && config.top_k == 0 && config.top_p >= 1.0f)
        return greedy(logits);
    if (config.temperature < 0.0f) throw std::invalid_argument("temperature must be >= 0");
    if (!(config.top_p > 0.0f && config.top_p <= 1.0f))
        throw std::invalid_argument("top_p must be in (0, 1]");

    const float temperature = config.temperature > 0.0f ? config.temperature : 1.0f;
    std::vector<std::size_t> indices(logits.size());
    for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
        return logits[a] > logits[b];
    });
    if (config.top_k > 0 && config.top_k < indices.size()) indices.resize(config.top_k);

    const float max_logit = logits[indices.front()];
    std::vector<double> weights;
    weights.reserve(indices.size());
    double total = 0.0;
    for (const auto i : indices) {
        const double w = std::exp((static_cast<double>(logits[i]) - max_logit) / temperature);
        weights.push_back(w);
        total += w;
    }

    if (config.top_p < 1.0f && weights.size() > 1) {
        std::size_t keep = 0;
        double cumulative = 0.0;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            cumulative += weights[i] / total;
            ++keep;
            if (cumulative >= config.top_p) break;
        }
        indices.resize(keep);
        weights.resize(keep);
    }

    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return indices[distribution(rng_)];
}

} // namespace swarx
