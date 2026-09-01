#pragma once

#include "swarx/online.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace swarx {

enum class IntelligenceIntent {
    General,
    TrendoraKnowledge,
    Time,
    Weather,
    LiveWeb,
    FinanceData,
    News,
    Calculator,
};

enum class SourceClass {
    OwnedKnowledge,
    LiveUtility,
    ExternalWeb,
    Computation,
    Unavailable,
};

struct IntelligenceRequest {
    std::string text;
    std::string location;
    std::string timezone = "UTC";
    double latitude = 0.0;
    double longitude = 0.0;
    bool network_allowed = true;
    bool require_fresh_information = false;
    std::size_t max_sources = 3;
};

struct RouteDecision {
    IntelligenceIntent intent = IntelligenceIntent::General;
    SourceClass source_class = SourceClass::Unavailable;
    std::string source_id;
    std::string rationale;
    bool requires_network = false;
    bool supported = false;
};

struct IntelligenceSource {
    std::string id;
    std::string name;
    SourceClass source_class = SourceClass::ExternalWeb;
    bool enabled = true;
    bool requires_network = true;
    int priority = 0;
};

struct IntelligenceResult {
    bool success = false;
    RouteDecision route;
    std::string answer_hint;
    std::string error;
    std::vector<SourceRecord> sources;
};

struct IntelligenceRouterConfig {
    bool network_enabled = true;
    bool allow_live_web = false;
    bool allow_finance_data = false;
    bool allow_news = false;
    OnlineConfig online{};
};

// Policy-aware decision layer above the HTTP gateway. Models request
// capabilities rather than arbitrary URLs.
class IntelligenceRouter {
public:
    explicit IntelligenceRouter(IntelligenceRouterConfig config = {});

    RouteDecision decide(const IntelligenceRequest& request) const;
    IntelligenceResult execute(const IntelligenceRequest& request) const;

    const std::vector<IntelligenceSource>& sources() const noexcept { return sources_; }

private:
    IntelligenceIntent classify(const std::string& text) const;
    RouteDecision route_for(IntelligenceIntent intent) const;

    IntelligenceRouterConfig config_;
    OnlineInformationGateway online_;
    std::vector<IntelligenceSource> sources_;
};

} // namespace swarx
