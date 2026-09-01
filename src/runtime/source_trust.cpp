#include "swarx/source_trust.hpp"

#include <algorithm>

namespace swarx {

SourceKind classify_source(const std::string& source_id) {
    if (source_id == "trendorafinds") return SourceKind::TrendoraFinds;
    if (source_id == "timeapi") return SourceKind::Time;
    if (source_id == "open-meteo") return SourceKind::Weather;
    if (source_id == "wikipedia") return SourceKind::Wikipedia;
    if (source_id == "web") return SourceKind::Web;
    if (source_id == "custom") return SourceKind::Custom;
    return SourceKind::Unknown;
}

TrustLevel trust_level(SourceKind kind) {
    switch (kind) {
        case SourceKind::TrendoraFinds:
        case SourceKind::Time:
        case SourceKind::Weather:
            return TrustLevel::Trusted;
        case SourceKind::Wikipedia:
        case SourceKind::Web:
            return TrustLevel::Known;
        case SourceKind::Custom:
            return TrustLevel::Unknown;
        case SourceKind::Unknown:
            return TrustLevel::Unknown;
    }
    return TrustLevel::Unknown;
}

double trust_score(SourceKind kind) {
    switch (kind) {
        case SourceKind::TrendoraFinds: return 0.95;
        case SourceKind::Time: return 0.94;
        case SourceKind::Weather: return 0.94;
        case SourceKind::Wikipedia: return 0.78;
        case SourceKind::Web: return 0.65;
        case SourceKind::Custom: return 0.35;
        case SourceKind::Unknown: return 0.0;
    }
    return 0.0;
}

bool accepts_evidence(const SourceEvidence& evidence, const SourcePolicy& policy) {
    if (evidence.status_code < 200 || evidence.status_code >= 300) return false;
    if (evidence.content.empty()) return false;
    if (evidence.trust == TrustLevel::Untrusted && !policy.allow_untrusted) return false;
    if (evidence.trust == TrustLevel::Unknown && !policy.allow_unknown) return false;
    return evidence.trust_score >= policy.minimum_trust_score;
}

std::vector<SourceEvidence> rank_evidence(std::vector<SourceEvidence> evidence,
                                           const SourcePolicy& policy) {
    evidence.erase(std::remove_if(evidence.begin(), evidence.end(),
                                  [&](const SourceEvidence& item) {
                                      return !accepts_evidence(item, policy);
                                  }),
                   evidence.end());
    std::stable_sort(evidence.begin(), evidence.end(),
                     [](const SourceEvidence& a, const SourceEvidence& b) {
                         if (a.trust_score != b.trust_score)
                             return a.trust_score > b.trust_score;
                         return a.content.size() > b.content.size();
                     });
    return evidence;
}

} // namespace swarx
