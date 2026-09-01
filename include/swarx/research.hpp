#pragma once

#include "swarx/evidence.hpp"
#include "swarx/knowledge.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace swarx {

enum class ResearchMode {
    Automatic,
    TrendoraFirst,
    BroadWeb,
    FreshnessCritical,
};

struct ResearchRequest {
    std::string query;
    std::string location;
    double latitude = 0.0;
    double longitude = 0.0;
    std::string timezone = "UTC";
    ResearchMode mode = ResearchMode::Automatic;
    bool allow_network = true;
    bool allow_external_sources = true;
    std::size_t max_sources = 4;
};

struct ResearchSourceSummary {
    KnowledgeRoute route = KnowledgeRoute::TrendoraFinds;
    bool attempted = false;
    bool succeeded = false;
    std::size_t evidence_count = 0;
    std::string error;
};

struct ResearchResult {
    bool success = false;
    bool requires_model_verification = false;
    std::string query;
    std::vector<ResearchSourceSummary> sources;
    EvidenceSet evidence{};
    std::string synthesis_hint;
};

struct ResearchConfig {
    KnowledgeFabricConfig knowledge{};
    EvidenceEngineConfig evidence{};
    std::size_t max_routes = 4;
};

// Coordinates multi-source retrieval without making any model provider part
// of the research layer. Retrieval is followed by provenance, trust, freshness
// and conflict evaluation before evidence is handed to a reasoning model.
class ResearchCoordinator {
public:
    explicit ResearchCoordinator(ResearchConfig config = {});

    ResearchResult research(const ResearchRequest& request) const;

private:
    static bool contains_any(const std::string& text, const char* const* terms, std::size_t count);
    static std::vector<KnowledgeRoute> select_routes(const ResearchRequest& request);

    ResearchConfig config_;
    KnowledgeFabric knowledge_;
    EvidenceEngine evidence_;
};

} // namespace swarx
