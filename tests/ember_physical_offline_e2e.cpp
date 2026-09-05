#include "swarx/resilience_gateway.hpp"
#include "swarx/web.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

std::string home_path(const std::string& relative) {
    const char* home = std::getenv("HOME");
    assert(home != nullptr);
    return std::string(home) + "/projects/swarx-ai-models/exports/swarx-ember-0.1/" + relative;
}

swarx::ResilienceGateway make_gateway() {
    swarx::ResilienceGatewayConfig config;
    config.cloud_enabled = true;
    config.ember_enabled = true;
    config.allow_network = true;
    config.retry_cloud_once = false;
    config.ember_runtime.tokenizer_vocab_path = home_path("tokenizer.vocab");

    swarx::ResilienceGateway gateway(config, [](std::string_view) {
        swarx::HttpClient client;
        swarx::HttpRequest request;
        request.url = "https://example.com/";
        request.timeout_seconds = 8;
        request.max_bytes = 64 * 1024;

        const auto http = client.get(request);
        swarx::CloudResponse response;
        response.success = http.ok();
        response.text = http.body;
        response.provider = "physical-http-test";
        response.model = "network-path-test";
        if (!response.success) {
            response.failure = swarx::GatewayFailure::Network;
            response.error = http.error.empty()
                ? "physical HTTP request failed"
                : http.error;
        }
        return response;
    });

    gateway.load_ember(home_path("model.swarx"));
    assert(gateway.ember_available());
    return gateway;
}

swarx::IntelligenceRequest make_request() {
    swarx::IntelligenceRequest request;
    request.text = "Search the web for the latest information about SWARX.";
    request.network_allowed = true;
    request.require_fresh_information = true;
    return request;
}

} // namespace

int main(int argc, char** argv) {
    using namespace swarx;

    if (argc != 2 || (std::string(argv[1]) != "online" &&
                      std::string(argv[1]) != "offline")) {
        std::cerr << "Usage: swarx-ember-physical-offline-e2e [online|offline]\n";
        return 2;
    }

    auto gateway = make_gateway();
    const auto result = gateway.chat(make_request());

    if (std::string(argv[1]) == "online") {
        assert(result.success);
        assert(result.path == IntelligencePath::Cloud);
        assert(!result.text.empty());
        assert(!result.ember_activated);
        std::cout << "SWARX EMBER physical E2E ONLINE: PASS\n";
        std::cout << "Cloud path reached through real HTTP transport.\n";
    } else {
        assert(result.success);
        assert(result.path == IntelligencePath::Ember);
        assert(result.ember_activated);
        assert(!result.text.empty());
        assert(result.error.find("Cloud unavailable") != std::string::npos);
        std::cout << "SWARX EMBER physical E2E OFFLINE: PASS\n";
        std::cout << "Real network failure triggered local EMBER fallback.\n";
    }

    std::cout << "EMBER response: " << result.text << "\n";
    return 0;
}
