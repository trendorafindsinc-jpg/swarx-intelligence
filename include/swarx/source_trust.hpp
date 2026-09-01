#pragma once

#include <string>
#include <vector>

namespace swarx {

enum class SourceKind {
    TrendoraFinds,
    Time,
    Weather,
    Wikipedia,
    Web,
    Custom,
    Unknown,
};

enum class TrustLevel {
    Trusted,
    Known,
    Unknown,
    Untrusted,
};

struct SourceEvidence {
    std::string source_id;
    std::string url;
    SourceKind kind = SourceKind::Unknown;
    TrustLevel trust = TrustLevel::Unknown;
    double trust_score = 0.0;
    std::string retrieved_at_utc;
    std::string published_at;
    long status_code = 0;
    std::string content_type;
    std::string content;
};

struct SourcePolicy {
    bool allow_unknown = true;
    bool allow_untrusted = false;
    double minimum_trust_score = 0.50;
};

SourceKind classify_source(const std::string& source_id);
TrustLevel trust_level(SourceKind kind);
double trust_score(SourceKind kind);

bool accepts_evidence(const SourceEvidence& evidence, const SourcePolicy& policy = {});

std::vector<SourceEvidence> rank_evidence(std::vector<SourceEvidence> evidence,
                                           const SourcePolicy& policy = {});

} // namespace swarx
