#pragma once

#include "swarx/model.hpp"
#include "swarx/sampler.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace swarx {

struct GenerationConfig {
    std::size_t max_new_tokens = 32;
    float temperature = 0.0f;          // <= 0 means greedy decoding.
    float repetition_penalty = 1.0f;   // 1.0 disables the penalty.
    std::uint32_t eos_token_id = 0;
    bool stop_on_eos = false;
    bool slide_context = true;
};

struct GenerationResult {
    std::vector<std::uint32_t> tokens;
    std::size_t generated_tokens = 0;
    bool stopped_on_eos = false;
    bool hit_context_limit = false;
};

using TokenCallback = std::function<void(std::uint32_t token)>;

class Generator {
public:
    explicit Generator(const TransformerModel& model, std::uint64_t seed = 0);

    GenerationResult generate(std::vector<std::uint32_t> prompt,
                              GenerationConfig config = {},
                              TokenCallback on_token = {}) const;

private:
    const TransformerModel& model_;
    std::uint64_t seed_;
};

} // namespace swarx
