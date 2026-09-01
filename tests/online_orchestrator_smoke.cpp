#include "swarx/online_orchestrator.hpp"

#include <cassert>
#include <chrono>
#include <iostream>

int main() {
    using namespace swarx;

    OnlineOrchestratorConfig config;
    config.knowledge.network_enabled = false;
    config.cache_ttl = std::chrono::seconds(30);

    OnlineIntelligenceOrchestrator orchestrator(config);

    OnlineOrchestrationRequest time_request;
    time_request.query = "What time is it in Lagos?";
    time_request.timezone = "Africa/Lagos";
    const auto time_plan = orchestrator.plan(time_request);
    assert(time_plan.need == OnlineNeed::Time);
    assert(time_plan.route == KnowledgeRoute::Time);
    assert(time_plan.requires_freshness);

    OnlineOrchestrationRequest weather_request;
    weather_request.query = "What is the weather forecast?";
    const auto weather_plan = orchestrator.plan(weather_request);
    assert(weather_plan.need == OnlineNeed::Weather);
    assert(weather_plan.route == KnowledgeRoute::Weather);
    assert(weather_plan.requires_freshness);

    OnlineOrchestrationRequest finance_request;
    finance_request.query = "Explain compound interest and savings.";
    const auto finance_plan = orchestrator.plan(finance_request);
    assert(finance_plan.need == OnlineNeed::Knowledge);
    assert(finance_plan.route == KnowledgeRoute::TrendoraFinds);
    assert(finance_plan.domain == KnowledgeDomain::Finance);

    OnlineOrchestrationRequest empty;
    const auto empty_result = orchestrator.execute(empty);
    assert(!empty_result.success);

    OnlineOrchestrationRequest blocked;
    blocked.query = "latest financial news";
    blocked.allow_network = false;
    blocked.allow_cache = false;
    const auto blocked_result = orchestrator.execute(blocked);
    assert(!blocked_result.success);
    assert(blocked_result.error.find("both network and cache") != std::string::npos);

    assert(orchestrator.cache_size() == 0);
    orchestrator.clear_cache();
    assert(orchestrator.cache_size() == 0);

    std::cout << "SWARX online orchestrator smoke test passed\n";
    return 0;
}
