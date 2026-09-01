#include "swarx/research.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>
#include <unordered_set>

namespace swarx {
namespace {

bool contains_ci(const std::string& text, const char* needle) {
    std::string a = text;
    std::string b = needle;
    std::transform(a.begin(), a.end(), a.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return a.find(b) != std::string::npos;
}

bool same_evidence(const KnowledgeEvidence& a, const KnowledgeEvidence& b) {
    return (!a.url.empty() && a.url == b.url) ||
           (a.source_id == b.source_id && a.content == b.content);
}

} // namespace

ResearchCoordinator::ResearchCoordinator(ResearchConfig config)
    : config_(std::move(config)), knowledge_(config_.knowledge), evidence_(config_.evidence) {}

bool ResearchCoordinator::contains_any(const std::string& text,
                                       const char* const* terms,
                                       std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (contains_ci(text, terms[i])) return true;
    }
    return false;
}

std::vector<KnowledgeRoute> ResearchCoordinator::select_routes(const ResearchRequest& request) {
    const std::string& q = request.query;
    std::vector<KnowledgeRoute> routes;

    const char* time_terms[] = {"time", "what time", "timezone", "local time"};
    const char* weather_terms[] = {"weather", "forecast", "temperature", "rain", "wind"};
    const char* finance_terms[] = {"finance", "money", "investment", "stock", "budget", "saving", "market"};

    if (contains_any(q, time_terms, 4)) routes.push_back(KnowledgeRoute::Time);
    if (contains_any(q, weather_terms, 5)) routes.push_back(KnowledgeRoute::Weather);

    if (request.mode == ResearchMode::TrendoraFirst ||
        (request.mode == ResearchMode::Automatic && contains_any(q, finance_terms, 7))) {
        routes.push_back(KnowledgeRoute::TrendoraFinds);
    }

    if (request.mode == ResearchMode::BroadWeb ||
        request.mode == ResearchMode::FreshnessCritical ||
        routes.empty()) {
        routes.push_back(KnowledgeRoute::GeneralWeb);
    }

    if (request.mode == ResearchMode::Automatic && routes.size() == 1 && routes.front() == KnowledgeRoute::TrendoraFinds) {
        routes.push_back(KnowledgeRoute::GeneralWeb);
    }

    std::vector<KnowledgeRoute> unique;
    for (const auto route : routes) {
        if (std::find(unique.begin(), unique.end(), route) == unique.end()) unique.push_back(route);
        if (unique.size() >= 4) break;
    }
    return unique;
}

ResearchResult ResearchCoordinator::research(const ResearchRequest& request) const {
    ResearchResult out;
    out.query = request.query;

    if (request.query.empty()) {
        out.synthesis_hint = "research request is empty";
        return out;
    }
    if (!request.allow_network || !config_.knowledge.network_enabled) {
        out.synthesis_hint = "network access is disabled; no online research performed";
        return out;
    }

    auto routes = select_routes(request);
    if (routes.size() > config_.max_routes) routes.resize(config_.max_routes);

    std::vector<KnowledgeEvidence> collected;
    std::unordered_set<std::string> seen;

    for (const auto route : routes) {
        ResearchSourceSummary summary;
        summary.route = route;
        summary.attempted = true;

        if (!request.allow_external_sources && route != KnowledgeRoute::TrendoraFinds) {
            summary.error = "external sources disabled by request";
            out.sources.push_back(std::move(summary));
            continue;
        }

        KnowledgeQuery query;
        query.text = request.query;
        query.route = route;
        query.location = request.location;
        query.latitude = request.latitude;
        query.longitude = request.longitude;
        query.timezone = request.timezone;
        query.max_sources = request.max_sources;
        query.require_fresh = request.mode == ResearchMode::FreshnessCritical ||
                              route == KnowledgeRoute::Time || route == KnowledgeRoute::Weather;

        const auto result = knowledge_.retrieve(query);
        summary.succeeded = result.success;
        summary.evidence_count = result.evidence.size();
        summary.error = result.error;
        out.sources.push_back(std::move(summary));

        if (!result.success) continue;
        for (const auto& item : result.evidence) {
            const std::string key = !item.url.empty() ? item.url : item.source_id + "\n" + item.content;
            if (seen.insert(key).second) collected.push_back(item);
        }
    }

    if (collected.empty()) {
        out.synthesis_hint = "no usable evidence was retrieved";
        return out;
    }

    // A second, content-level guard removes duplicate evidence that arrived
    // through different routes before scoring and conflict analysis.
    std::vector<KnowledgeEvidence> unique;
    for (const auto& item : collected) {
        const bool duplicate = std::any_of(unique.begin(), unique.end(),
            [&](const KnowledgeEvidence& existing) { return same_evidence(existing, item); });
        if (!duplicate) unique.push_back(item);
    }

    out.evidence = evidence_.evaluate(unique, request.query);
    out.success = !out.evidence.items.empty();
    out.requires_model_verification = out.evidence.status == EvidenceStatus::Conflicting ||
                                       out.evidence.confidence < 0.70;

    if (out.evidence.status == EvidenceStatus::Conflicting) {
        out.synthesis_hint = "multiple claims require model-level verification before a definitive answer";
    } else if (out.evidence.confidence < 0.70) {
        out.synthesis_hint = "evidence is usable but confidence is limited; qualify the answer";
    } else {
        out.synthesis_hint = "evidence is ranked and ready for model synthesis";
    }
    return out;
}

} // namespace swarx
