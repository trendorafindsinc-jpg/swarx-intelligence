#pragma once

#include "swarx/sampler.hpp"
#include "swarx/tokenizer.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace swarx {

struct EngineConfig {
    std::size_t context_size = 2048;
    float temperature = 0.7f;
};

class Engine {
public:
    explicit Engine(EngineConfig config = {});

    const EngineConfig& config() const noexcept { return config_; }
    std::vector<TokenId> tokenize(std::string_view text) const;

    // Runtime bootstrap endpoint. Model execution will plug into this API
    // without coupling callers to a specific backend implementation.
    std::string generate(std::string_view prompt, std::size_t max_tokens = 128);

private:
    EngineConfig config_;
    Tokenizer tokenizer_;
    Sampler sampler_;
};

} // namespace swarx
