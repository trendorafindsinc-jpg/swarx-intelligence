#include "swarx/intelligence_bus.hpp"

#include <cassert>
#include <iostream>

int main() {
    using namespace swarx;

    IntelligenceBus bus;
    const auto request_id = bus.begin_request("Research a current finance question");
    assert(request_id != 0);

    IntelligenceEnvelope observation;
    observation.request_id = request_id;
    observation.kind = BusMessageKind::Observation;
    observation.sender = "trendora-tool";
    observation.task = "Research a current finance question";
    observation.payload = "retrieved knowledge";
    observation.source_ids = {"trendora"};
    observation.confidence = 0.9;
    assert(bus.publish(observation));

    IntelligenceEnvelope response;
    response.request_id = request_id;
    response.parent_id = request_id;
    response.kind = BusMessageKind::ModelResponse;
    response.sender = "swarx-core";
    response.task = "Research a current finance question";
    response.payload = "draft answer";
    response.source_ids = {"trendora"};
    response.confidence = 0.8;
    assert(bus.publish(response));

    IntelligenceEnvelope invalid = response;
    invalid.confidence = 1.5;
    assert(!bus.publish(invalid));

    const auto messages = bus.messages(request_id);
    assert(messages.size() == 3);

    const auto trace = bus.trace(request_id);
    assert(trace.message_count == 3);
    assert(trace.participants.size() == 2);
    assert(trace.sources.size() == 1);
    assert(trace.sources.front() == "trendora");
    assert(!trace.has_error);

    bus.clear();
    assert(bus.messages(request_id).empty());

    std::cout << "SWARX intelligence bus smoke test passed\n";
    return 0;
}
