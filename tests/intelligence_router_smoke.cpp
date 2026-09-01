#include "swarx/intelligence_router.hpp"
#include "swarx/tool_registry.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>

int main() {
    using namespace swarx;

    IntelligenceRouter router;

    IntelligenceRequest trendora;
    trendora.text = "Explain compound interest using TrendoraFinds";
    auto decision = router.decide(trendora);
    assert(decision.intent == IntelligenceIntent::TrendoraKnowledge);
    assert(decision.source_id == "trendora");
    assert(decision.supported);

    IntelligenceRequest time;
    time.text = "What time is it in Lagos?";
    time.timezone = "Africa/Lagos";
    decision = router.decide(time);
    assert(decision.intent == IntelligenceIntent::Time);
    assert(decision.source_id == "timeapi");
    assert(decision.requires_network);

    IntelligenceRequest weather;
    weather.text = "What's the weather forecast?";
    weather.latitude = 6.5244;
    weather.longitude = 3.3792;
    weather.timezone = "Africa/Lagos";
    decision = router.decide(weather);
    assert(decision.intent == IntelligenceIntent::Weather);
    assert(decision.source_id == "open-meteo");

    IntelligenceRequest offline = time;
    offline.network_allowed = false;
    decision = router.decide(offline);
    assert(!decision.supported);

    IntelligenceRouterConfig restricted_config;
    restricted_config.allow_live_web = false;
    restricted_config.allow_finance_data = false;
    restricted_config.allow_news = false;
    IntelligenceRouter restricted(restricted_config);

    IntelligenceRequest current;
    current.text = "Search online for the latest news";
    decision = restricted.decide(current);
    assert(decision.intent == IntelligenceIntent::News);
    assert(!decision.supported);

    ToolRegistry tools;
    assert(tools.available(ToolCapability::Calculator));
    assert(tools.available(ToolCapability::Time));
    assert(tools.available(ToolCapability::Weather));
    assert(!tools.available(ToolCapability::WebSearch));

    if (std::getenv("SWARX_RUN_ONLINE_TESTS") != nullptr) {
        const auto live = router.execute(time);
        assert(live.success);
        assert(!live.sources.empty());
        assert(live.sources.front().source_id == "timeapi");
    }

    std::cout << "SWARX intelligence router smoke test passed\n";
    return 0;
}
