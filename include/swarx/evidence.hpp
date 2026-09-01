#pragma once

#include "swarx/knowledge.hpp"
#include "swarx/source_trust.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace swarx {

enum class EvidenceStatus {
    Empty,
    Ready,
    Conflicting,
    Rejected,
};

struct EvidenceItem {
    KnowledgeEvidence evidence{};
    double trust_score = 0.0;
    double freshness_score = 0.0;
    double relevance_score = 0.0;
    double composite_score = 0.0;
};

struct EvidenceSet {
    EvidenceStatus status = EvidenceStatus::Empty;
    std::vector<EvidenceItem> items;
    std::vector<std::string> conflicts;
    double confidence = 0.0;
    bool provenance_complete = false;
    std::string summary;
};

struct EvidenceEngineConfig {
    SourcePolicy source_policy{};
    std::size_t max_items = 8;
    bool require_provenance = true;
    bool detect_conflicts = true;
};

// Converts raw knowledge evidence into a bounded, trusted evidence set.
// It scores trust, freshness and relevance, rejects disallowed sources,
// detects materially different claims, and reports a conservative confidence.
class EvidenceEngine {
public:
    explicit EvidenceEngine(EvidenceEngineConfig config = {});

    EvidenceSet evaluate(const std::vector<KnowledgeEvidence>& evidence,
                         const std::string& query = {}) const;

private:
    static std::string normalize(const std::string& text);
    static double freshness_score(Freshness freshness);
    static double relevance_score(const KnowledgeEvidence& evidence, const std::string& query);
    static std::string claim_key(const KnowledgeEvidence& evidence);

    EvidenceEngineConfig config_;
};

} // namespace swarx
