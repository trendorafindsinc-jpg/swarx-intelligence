#pragma once

#include "swarx/online.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace swarx {

enum class KnowledgeDomain { General, Finance, Education, Market, Product };

enum class Freshness { Unknown, Stale, Recent, Live };

struct KnowledgeQuery {
    std::string text;
    KnowledgeDomain domain = KnowledgeDomain::General;
    bool require_fresh = false;
    std::size_t max_sources = 3;
};

struct KnowledgeSource {
    std::string id;
    std::string name;
    std::string url;
    int priority = 0;
    bool owned = false;
    bool enabled = true;
};

struct KnowledgeEvidence {
    std::string source_id;
    std::string title;
    std::string url;
    std::string content;
    std::string retrieved_at_utc;
    Freshness freshness = Freshness::Unknown;
    int reliability = 0;
};

struct KnowledgeResult {
    bool success = false;
    std::string error;
    std::string query;
    std::vector<KnowledgeEvidence> evidence;
};

struct KnowledgeFabricConfig {
    OnlineConfig online{};
    bool network_enabled = true;
    bool allow_external_sources = true;
    bool require_provenance = true;
};

class KnowledgeFabric {
public:
    explicit KnowledgeFabric(KnowledgeFabricConfig config = {});

    KnowledgeResult retrieve(const KnowledgeQuery& query) const;
    const std::vector<KnowledgeSource>& sources() const noexcept { return sources_; }

private:
    KnowledgeDomain classify_domain(const std::string& text) const;
    KnowledgeResult retrieve_trendorafinds(const KnowledgeQuery& query) const;

    KnowledgeFabricConfig config_;
    OnlineInformationGateway online_;
    std::vector<KnowledgeSource> sources_;
};

} // namespace swarx
