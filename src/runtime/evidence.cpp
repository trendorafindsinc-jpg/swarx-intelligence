#include "swarx/evidence.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace swarx {
namespace {

SourceEvidence to_source_evidence(const KnowledgeEvidence& item) {
    SourceEvidence out;
    out.source_id = item.source_id;
    out.url = item.url;
    out.kind = classify_source(item.source_id);
    out.trust = trust_level(out.kind);
    out.trust_score = item.reliability > 0 ? std::min(1.0, item.reliability / 100.0) : swarx::trust_score(out.kind);
    out.retrieved_at_utc = item.retrieved_at_utc;
    out.content = item.content;
    return out;
}

} // namespace

EvidenceEngine::EvidenceEngine(EvidenceEngineConfig config) : config_(std::move(config)) {}

std::string EvidenceEngine::normalize(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool previous_space = false;
    for (unsigned char c : text) {
        if (std::isspace(c)) {
            if (!out.empty()) previous_space = true;
            continue;
        }
        if (previous_space) out.push_back(' ');
        previous_space = false;
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

double EvidenceEngine::freshness_score(Freshness freshness) {
    switch (freshness) {
        case Freshness::Live: return 1.0;
        case Freshness::Recent: return 0.85;
        case Freshness::Stale: return 0.35;
        case Freshness::Unknown: return 0.50;
    }
    return 0.0;
}

double EvidenceEngine::relevance_score(const KnowledgeEvidence& evidence, const std::string& query) {
    if (query.empty()) return 0.75;
    const std::string haystack = normalize(evidence.title + " " + evidence.content);
    const std::string needle = normalize(query);
    if (needle.empty()) return 0.75;
    if (haystack.find(needle) != std::string::npos) return 1.0;

    std::istringstream words(needle);
    std::size_t total = 0;
    std::size_t hits = 0;
    std::string word;
    while (words >> word) {
        if (word.size() < 2) continue;
        ++total;
        if (haystack.find(word) != std::string::npos) ++hits;
    }
    return total == 0 ? 0.50 : static_cast<double>(hits) / static_cast<double>(total);
}

std::string EvidenceEngine::claim_key(const KnowledgeEvidence& evidence) {
    return normalize(evidence.content);
}

EvidenceSet EvidenceEngine::evaluate(const std::vector<KnowledgeEvidence>& evidence,
                                      const std::string& query) const {
    EvidenceSet out;
    if (evidence.empty()) {
        out.summary = "no evidence supplied";
        return out;
    }

    std::vector<SourceEvidence> sources;
    sources.reserve(evidence.size());
    for (const auto& item : evidence) sources.push_back(to_source_evidence(item));
    sources = rank_evidence(std::move(sources), config_.source_policy);

    std::unordered_set<std::string> claims;
    std::vector<EvidenceItem> candidates;
    candidates.reserve(sources.size());

    for (const auto& source : sources) {
        if (candidates.size() >= config_.max_items) break;
        if (!accepts_evidence(source, config_.source_policy)) continue;

        const auto it = std::find_if(evidence.begin(), evidence.end(), [&](const KnowledgeEvidence& item) {
            return item.source_id == source.source_id && item.url == source.url && item.content == source.content;
        });
        if (it == evidence.end()) continue;

        EvidenceItem item;
        item.evidence = *it;
        item.trust_score = source.trust_score;
        item.freshness_score = freshness_score(it->freshness);
        item.relevance_score = relevance_score(*it, query);
        item.composite_score = 0.50 * item.trust_score +
                               0.30 * item.freshness_score +
                               0.20 * item.relevance_score;
        candidates.push_back(std::move(item));
        claims.insert(claim_key(*it));
    }

    std::sort(candidates.begin(), candidates.end(), [](const EvidenceItem& a, const EvidenceItem& b) {
        return a.composite_score > b.composite_score;
    });
    out.items = std::move(candidates);

    out.provenance_complete = config_.require_provenance;
    for (const auto& item : out.items) {
        if (item.evidence.source_id.empty() || item.evidence.url.empty()) {
            out.provenance_complete = false;
            break;
        }
    }

    if (out.items.empty()) {
        out.status = EvidenceStatus::Rejected;
        out.summary = "all supplied evidence was rejected by source policy";
        return out;
    }

    if (config_.detect_conflicts && claims.size() > 1) {
        out.status = EvidenceStatus::Conflicting;
        out.conflicts.push_back("sources contain materially different claims");
    } else {
        out.status = EvidenceStatus::Ready;
    }

    double weighted = 0.0;
    double weight_sum = 0.0;
    for (const auto& item : out.items) {
        const double weight = std::max(0.01, item.composite_score);
        weighted += item.composite_score * weight;
        weight_sum += weight;
    }
    out.confidence = weight_sum == 0.0 ? 0.0 : std::clamp(weighted / weight_sum, 0.0, 1.0);
    if (out.status == EvidenceStatus::Conflicting) out.confidence *= 0.65;
    if (!out.provenance_complete) out.confidence *= 0.75;

    out.summary = out.status == EvidenceStatus::Conflicting
        ? "evidence collected, but claims conflict"
        : "evidence collected and ranked";
    return out;
}

} // namespace swarx
