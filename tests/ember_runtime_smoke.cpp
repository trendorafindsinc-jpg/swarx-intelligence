#include "swarx/resilience_gateway.hpp"
#include "swarx/runtime.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

int main() {
    using namespace swarx;

    const std::string model_path =
        std::string(std::getenv("HOME")) +
        "/projects/swarx-ai-models/exports/swarx-ember-0.1/model.swarx";

    const std::string tokenizer_path =
        std::string(std::getenv("HOME")) +
        "/projects/swarx-ai-models/exports/swarx-ember-0.1/tokenizer.vocab";

    RuntimeConfig runtime_config;
    runtime_config.tokenizer_vocab_path = tokenizer_path;

    Runtime runtime(runtime_config);
    runtime.load(model_path);

    // Runtime health.
    assert(runtime.loaded());
    assert(runtime.health().model_loaded);
    assert(runtime.health().model_valid);
    assert(runtime.health().finite_parameters);
    assert(runtime.health().parameter_count == 541312);
    assert(runtime.health().vocabulary_size == 512);
    assert(runtime.health().context_length == 128);

    // Tokenizer compatibility.
    assert(runtime.chat_session().inference().tokenizer_loaded());
    assert(runtime.chat_session().inference().tokenizer_vocab_size() == 512);

    // Real local generation.
    IntelligenceRequest request;
    request.text = "Hello";

    auto local_result = runtime.chat(request.text);

    assert(local_result.message.content.size() > 0);

    // Real EMBER fallback through the resilience gateway.
    ResilienceGatewayConfig gateway_config;
    gateway_config.cloud_enabled = true;
    gateway_config.allow_network = true;
    gateway_config.ember_enabled = true;
    gateway_config.retry_cloud_once = false;
    gateway_config.ember_runtime = runtime_config;

    ResilienceGateway gateway(gateway_config, [](std::string_view) {
        CloudResponse response;
        response.success = false;
        response.failure = GatewayFailure::ProviderFailure;
        response.error = "simulated cloud outage";
        return response;
    });

    gateway.load_ember(model_path);

    auto fallback_result = gateway.chat(request);

    assert(fallback_result.success);
    assert(fallback_result.path == IntelligencePath::Ember);
    assert(fallback_result.ember_activated);
    assert(!fallback_result.text.empty());

    std::cout << "SWARX EMBER-0.1 runtime smoke test passed\n";
    std::cout << "Parameters: "
              << runtime.health().parameter_count << "\n";
    std::cout << "Vocabulary: "
              << runtime.health().vocabulary_size << "\n";
    std::cout << "Local response: "
              << local_result.message.content << "\n";
    std::cout << "Fallback response: "
              << fallback_result.text << "\n";

    return 0;
}
