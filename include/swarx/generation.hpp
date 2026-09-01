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
    float temperature = 0.0f;
    std::size_t top_k = 0;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    std::uint32_t eos_token_id = 0;
    bool stop_on_eos = false;
    bool slide_context = true;
};

struct GenerationResult {
    std::vector<std::uint32_t> tokens;
    std::size_t generated_tokens = 0;
    bool stopped_on_eos = false;
    bool hit_context_limit = false;
    bool cancelled = false;
};

using TokenCallback = std::function<void(std::uint32_t token)>;
using CancelCallback = std::function<bool()>;

class Generator {
public:
    explicit Generator(const TransformerModel& model, std::uint64_t seed = 0);

    GenerationResult generate(std::vector<std::uint32_t> prompt,
                              GenerationConfig config = {},
                              TokenCallback on_token = {},
                              CancelCallback should_cancel = {}) const;

private:
    const TransformerModel& model_;
    std::uint64_t seed_;
};

} // namespace swarx
