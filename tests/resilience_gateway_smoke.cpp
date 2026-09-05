#include "swarx/resilience_gateway.hpp"

#include <cassert>
#include <iostream>

int main() {
    using namespace swarx;

    int cloud_calls = 0;
    ResilienceGatewayConfig config;
    config.ember_enabled = true;
    config.cloud_enabled = true;
    config.allow_network = true;
    config.retry_cloud_once = true;

    ResilienceGateway gateway(config, [&](std::string_view message) {
        ++cloud_calls;
        CloudResponse response;
        response.success = true;
        response.text = std::string("cloud: ") + std::string(message);
        response.provider = "test-provider";
        response.model = "test-model";
        return response;
    });

    IntelligenceRequest request;
    request.text = "hello";
    auto result = gateway.chat(request);
    assert(result.success);
    assert(result.path == IntelligencePath::Cloud);
    assert(result.text == "cloud: hello");
    assert(cloud_calls == 1);

    ResilienceGatewayConfig offline_config;
    offline_config.cloud_enabled = false;
    offline_config.allow_network = false;
    offline_config.ember_enabled = true;
    ResilienceGateway offline(offline_config);

    result = offline.chat(request);
    assert(!result.success);
    assert(result.path == IntelligencePath::Unavailable);
    assert(result.failure == GatewayFailure::ModelUnavailable);

    int failed_calls = 0;
    ResilienceGatewayConfig failover_config;
    failover_config.cloud_enabled = true;
    failover_config.allow_network = true;
    failover_config.ember_enabled = true;
    failover_config.retry_cloud_once = true;
    ResilienceGateway failover(failover_config, [&](std::string_view) {
        ++failed_calls;
        CloudResponse response;
        response.success = false;
        response.failure = GatewayFailure::ProviderFailure;
        response.error = "simulated provider outage";
        return response;
    });

    result = failover.chat(request);
    assert(!result.success);
    assert(result.path == IntelligencePath::Cloud);
    assert(failed_calls == 2);
    assert(result.error.find("simulated provider outage") != std::string::npos);

    std::cout << "SWARX EMBER resilience gateway smoke test passed\n";
    return 0;
}
