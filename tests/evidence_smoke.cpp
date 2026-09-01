#include "swarx/evidence.hpp"

#include <cassert>
#include <iostream>

int main() {
    using namespace swarx;

    EvidenceEngine engine;
    KnowledgeEvidence first;
    first.source_id = "trendorafinds";
    first.url = "https://example.test/trendora";
    first.title = "Finance guide";
    first.content = "Saving money requires a budget.";
    first.freshness = Freshness::Recent;
    first.reliability = 100;

    KnowledgeEvidence second = first;
    second.source_id = "wikipedia";
    second.url = "https://example.test/wiki";
    second.reliability = 70;

    auto result = engine.evaluate({first, second}, "saving money");
    assert(result.status == EvidenceStatus::Ready);
    assert(result.items.size() == 2);
    assert(result.provenance_complete);
    assert(result.confidence > 0.0);

    KnowledgeEvidence conflicting = second;
    conflicting.content = "Saving money requires no budget.";
    auto conflict = engine.evaluate({first, conflicting}, "saving money");
    assert(conflict.status == EvidenceStatus::Conflicting);
    assert(!conflict.conflicts.empty());
    assert(conflict.confidence < result.confidence);

    KnowledgeEvidence missing = first;
    missing.url.clear();
    auto rejected = engine.evaluate({missing});
    assert(rejected.status == EvidenceStatus::Rejected || !rejected.provenance_complete);

    std::cout << "evidence smoke tests passed\n";
    return 0;
}
