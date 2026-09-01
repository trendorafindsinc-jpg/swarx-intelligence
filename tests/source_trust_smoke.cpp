#include "swarx/source_trust.hpp"

#include <cassert>
#include <iostream>

int main() {
    using namespace swarx;

    assert(classify_source("trendorafinds") == SourceKind::TrendoraFinds);
    assert(classify_source("open-meteo") == SourceKind::Weather);
    assert(trust_level(SourceKind::TrendoraFinds) == TrustLevel::Trusted);
    assert(trust_score(SourceKind::TrendoraFinds) > trust_score(SourceKind::Wikipedia));

    SourceEvidence good;
    good.source_id = "trendorafinds";
    good.url = "https://trendorafinds.wordpress.com/";
    good.kind = classify_source(good.source_id);
    good.trust = trust_level(good.kind);
    good.trust_score = trust_score(good.kind);
    good.status_code = 200;
    good.content = "financial education";
    assert(accepts_evidence(good));

    SourceEvidence bad = good;
    bad.status_code = 500;
    assert(!accepts_evidence(bad));

    SourceEvidence unknown = good;
    unknown.source_id = "custom";
    unknown.kind = SourceKind::Custom;
    unknown.trust = TrustLevel::Unknown;
    unknown.trust_score = trust_score(SourceKind::Custom);
    assert(!accepts_evidence(unknown));

    auto ranked = rank_evidence({unknown, good});
    assert(ranked.size() == 1);
    assert(ranked.front().source_id == "trendorafinds");

    SourcePolicy strict;
    strict.allow_unknown = false;
    strict.minimum_trust_score = 0.90;
    assert(accepts_evidence(good, strict));

    std::cout << "SWARX source trust smoke test passed\n";
    return 0;
}
