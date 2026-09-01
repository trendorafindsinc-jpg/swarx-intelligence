#include "swarx/knowledge.hpp"

#include "swarx/source_adapters.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <utility>

namespace swarx {
namespace {

bool contains_ci(const std::string& text, const std::string& needle) {
    std::string a = text;
    std::string b = needle;
    std::transform(a.begin(), a.end(), a.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return a.find(b) != std::string::npos;
}

std::string now_utc() {
    const auto now = std::chrono::system_clock::now();
    const auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

} // namespace

KnowledgeFabric::KnowledgeFabric(KnowledgeFabricConfig config)
    : config_(std::move(config)), registry_(make_default_source_registry(config_.online)) {
    sources_.push_back({"trendorafinds", "TrendoraFinds", config_.online.trendora_endpoint, 100, true, true});
    sources_.push_back({"timeapi", "TimeAPI", config_.online.time_endpoint, 95, false, true});
    sources_.push_back({"open-meteo", "Open-Meteo", config_.online.weather_endpoint, 95, false, true});
    sources_.push_back({"wikipedia", "Wikipedia", config_.online.wikipedia_endpoint, 70, false, true});
}

KnowledgeDomain KnowledgeFabric::classify_domain(const std::string& text) const {
    if (contains_ci(text, "finance") || contains_ci(text, "money") || contains_ci(text, "investment") ||
        contains_ci(text, "stock") || contains_ci(text, "budget") || contains_ci(text, "saving")) {
        return KnowledgeDomain::Finance;
    }
    if (contains_ci(text, "learn") || contains_ci(text, "explain") || contains_ci(text, "education") ||
        contains_ci(text, "lesson") || contains_ci(text, "study")) {
        return KnowledgeDomain::Education;
    }
    if (contains_ci(text, "market") || contains_ci(text, "price")) return KnowledgeDomain::Market;
    return KnowledgeDomain::General;
}

KnowledgeRoute KnowledgeFabric::route_for_capability(SourceCapability capability) {
    switch (capability) {
        case SourceCapability::TrendoraFinds: return KnowledgeRoute::TrendoraFinds;
        case SourceCapability::Time: return KnowledgeRoute::Time;
        case SourceCapability::Weather: return KnowledgeRoute::Weather;
        case SourceCapability::GeneralWeb: return KnowledgeRoute::GeneralWeb;
    }
    return KnowledgeRoute::GeneralWeb;
}

KnowledgeResult KnowledgeFabric::retrieve(const KnowledgeQuery& query) const {
    KnowledgeQuery normalized = query;
    if (normalized.domain == KnowledgeDomain::General) normalized.domain = classify_domain(query.text);
    if (!config_.network_enabled) return {false, "network access disabled by policy", query.text, normalized.route, {}};

    SourceCapability capability = SourceCapability::TrendoraFinds;
    switch (normalized.route) {
        case KnowledgeRoute::TrendoraFinds: capability = SourceCapability::TrendoraFinds; break;
        case KnowledgeRoute::Time: capability = SourceCapability::Time; break;
        case KnowledgeRoute::Weather: capability = SourceCapability::Weather; break;
        case KnowledgeRoute::GeneralWeb: capability = SourceCapability::GeneralWeb; break;
    }

    if (!config_.allow_external_sources && capability != SourceCapability::TrendoraFinds) {
        return {false, "external knowledge sources disabled by policy", query.text, normalized.route, {}};
    }
    return retrieve_online(normalized, capability);
}

KnowledgeResult KnowledgeFabric::retrieve_online(const KnowledgeQuery& query, SourceCapability capability) const {
    const auto source = registry_.find(capability);
    KnowledgeResult out;
    out.query = query.text;
    out.route = route_for_capability(capability);

    if (!source) {
        out.error = "no registered source for requested capability";
        return out;
    }

    SourceRequest request;
    request.capability = capability;
    request.query = query.text;
    request.location = query.location;
    request.latitude = query.latitude;
    request.longitude = query.longitude;
    request.timezone = query.timezone;
    request.policy.minimum_trust_score = config_.allow_external_sources ? 0.50 : 1.0;

    const auto result = source->fetch(request);
    if (!result.success) {
        out.error = result.error.empty() ? "knowledge source request failed" : result.error;
        return out;
    }

    KnowledgeEvidence evidence;
    evidence.source_id = result.source_id;
    evidence.url = result.url;
    evidence.content = result.body;
    evidence.retrieved_at_utc = result.evidence.retrieved_at_utc.empty() ? now_utc() : result.evidence.retrieved_at_utc;
    evidence.reliability = static_cast<int>(result.evidence.trust_score * 100.0);
    evidence.freshness = (query.route == KnowledgeRoute::Time || query.route == KnowledgeRoute::Weather)
                             ? Freshness::Live
                             : (query.require_fresh ? Freshness::Recent : Freshness::Unknown);

    if (config_.require_provenance && (evidence.source_id.empty() || evidence.url.empty())) {
        out.error = "knowledge result missing required provenance";
        return out;
    }

    out.evidence.push_back(std::move(evidence));
    out.success = !out.evidence.empty();
    return out;
}

} // namespace swarx
