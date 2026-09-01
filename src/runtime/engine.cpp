#include "swarx/engine.hpp"

#include <sstream>

namespace swarx {

Engine::Engine(EngineConfig config) : config_(config), sampler_(42) {}

std::vector<TokenId> Engine::tokenize(std::string_view text) const {
    return tokenizer_.encode(text);
}

std::string Engine::generate(std::string_view prompt, std::size_t max_tokens) {
    // Deliberately conservative bootstrap behavior: the engine is wired and
    // testable before a heavyweight model backend is introduced.
    // This prevents the public API from being tied to llama.cpp or another
    // third-party runtime.
    (void)max_tokens;
    std::ostringstream out;
    out << "SWARX Intelligence runtime is ready. Prompt received: " << prompt;
    return out.str();
}

} // namespace swarx
