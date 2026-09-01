#include "swarx/research.hpp"

#include <cassert>

int main() {
    swarx::ResearchConfig config;
    config.knowledge.network_enabled = false;
    swarx::ResearchCoordinator coordinator(config);

    swarx::ResearchRequest empty;
    auto result = coordinator.research(empty);
    assert(!result.success);
    assert(result.evidence.items.empty());

    swarx::ResearchRequest blocked;
    blocked.query = "What is the weather today?";
    auto blocked_result = coordinator.research(blocked);
    assert(!blocked_result.success);
    assert(blocked_result.synthesis_hint.find("disabled") != std::string::npos);

    return 0;
}
