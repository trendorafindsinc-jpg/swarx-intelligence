#include "swarx/knowledge.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <sstream>

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
#if defined(_WIN32)
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

} // namespace

KnowledgeFabric::KnowledgeFabric(KnowledgeFabricConfig config)
    : config_(std::move(config)), online_(config_.online) {
    sources_.push_back({"trendorafinds", "TrendoraFinds", config_.online.trendora_endpoint, 100, true, true});
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

KnowledgeResult KnowledgeFabric::retrieve(const KnowledgeQuery& query) const {
    KnowledgeQuery normalized = query;
    if (normalized.domain == KnowledgeDomain::General) normalized.domain = classify_domain(query.text);
    if (!config_.network_enabled) return {false, "network access disabled by policy", query.text, {}};
    if (!config_.allow_external_sources && normalized.domain == KnowledgeDomain::General)
        return retrieve_trendorafinds(normalized);
    return retrieve_trendorafinds(normalized);
}

KnowledgeResult KnowledgeFabric::retrieve_trendorafinds(const KnowledgeQuery& query) const {
    OnlineRequest request;
    request.intent = OnlineIntent::Trendora;
    request.query = query.text;
    const auto result = online_.fetch(request);
    KnowledgeResult out;
    out.query = query.text;
    if (!result.success) {
        out.error = result.error;
        return out;
    }
    KnowledgeEvidence evidence;
    evidence.source_id = result.source.source_id;
    evidence.url = result.source.url;
    evidence.content = result.source.body;
    evidence.retrieved_at_utc = result.source.retrieved_at_utc.empty() ? now_utc() : result.source.retrieved_at_utc;
    evidence.freshness = query.require_fresh ? Freshness::Recent : Freshness::Unknown;
    evidence.reliability = 100;
    out.evidence.push_back(std::move(evidence));
    out.success = !out.evidence.empty();
    return out;
}

} // namespace swarx
