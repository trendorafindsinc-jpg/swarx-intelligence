#pragma once

#include "swarx/knowledge.hpp"

#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

namespace swarx {

enum class OnlineNeed {
    None,
    Knowledge,
    Time,
    Weather,
    GeneralWeb,
};

struct OnlinePlan {
    OnlineNeed need = OnlineNeed::None;
    KnowledgeRoute route = KnowledgeRoute::TrendoraFinds;
    KnowledgeDomain domain = KnowledgeDomain::General;
    bool requires_freshness = false;
    std::string reason;
};

struct OnlineOrchestrationRequest {
    std::string query;
    std::string location;
    double latitude = 0.0;
    double longitude = 0.0;
    std::string timezone = "UTC";
    bool allow_network = true;
    bool allow_cache = true;
    std::size_t max_sources = 3;
};

struct OnlineOrchestrationResult {
    bool success = false;
    bool from_cache = false;
    OnlinePlan plan{};
    KnowledgeResult knowledge{};
    std::string error;
};

struct OnlineOrchestratorConfig {
    KnowledgeFabricConfig knowledge{};
    std::chrono::seconds cache_ttl{30};
    std::size_t max_cache_entries = 64;
};

// Turns a natural-language request into a controlled online-information plan,
// executes it through the KnowledgeFabric, and optionally serves recent results
// from a bounded in-memory cache. It never exposes provider URLs to callers.
class OnlineIntelligenceOrchestrator {
public:
    explicit OnlineIntelligenceOrchestrator(OnlineOrchestratorConfig config = {});

    OnlinePlan plan(const OnlineOrchestrationRequest& request) const;
    OnlineOrchestrationResult execute(const OnlineOrchestrationRequest& request) const;
    void clear_cache() const;
    std::size_t cache_size() const;

private:
    struct CacheEntry {
        KnowledgeResult result;
        std::chrono::steady_clock::time_point expires_at;
    };

    static std::string normalize(const std::string& text);
    static bool contains_any(const std::string& text, const char* const* terms, std::size_t count);
    static std::string cache_key(const OnlineOrchestrationRequest& request, const OnlinePlan& plan);
    void trim_cache_locked() const;

    OnlineOrchestratorConfig config_;
    KnowledgeFabric knowledge_;
    mutable std::mutex cache_mutex_;
    mutable std::unordered_map<std::string, CacheEntry> cache_;
};

} // namespace swarx
