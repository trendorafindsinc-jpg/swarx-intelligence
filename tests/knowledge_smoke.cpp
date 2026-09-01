#include "swarx/knowledge.hpp"

#include <cassert>
#include <iostream>

int main() {
    swarx::KnowledgeFabricConfig config;
    config.network_enabled = false;
    swarx::KnowledgeFabric fabric(config);

    assert(!fabric.sources().empty());
    assert(fabric.sources().front().id == "trendorafinds");
    assert(fabric.sources().front().owned);

    swarx::KnowledgeQuery query;
    query.text = "explain personal finance and saving";
    auto result = fabric.retrieve(query);
    assert(!result.success);
    assert(!result.error.empty());

    std::cout << "knowledge fabric policy/provenance smoke passed\n";
    return 0;
}
