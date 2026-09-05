#include "swarx/resilience_gateway.hpp"

#include <stdexcept>
#include <utility>

namespace swarx {

ResilienceGateway::ResilienceGateway(ResilienceGatewayConfig config)
    : config_(std::move(config)), ember_(config_.ember_runtime) {}

ResilienceGateway::ResilienceGateway(ResilienceGatewayConfig config, CloudAdapter cloud)
    : config_(std::move(config)), cloud_(std::move(cloud)), ember_(config_.ember_runtime) {}

void ResilienceGateway::set_cloud_adapter(CloudAdapter cloud) {
    cloud_ = std::move(cloud);
}

void ResilienceGateway::set_ember_runtime(Runtime runtime) {
    ember_ = std::move(runtime);
}

void ResilienceGateway::load_ember(const std::string& weights_path) {
    ember_.load(weights_path);
}

bool ResilienceGateway::ember_available() const noexcept {
    return config_.ember_enabled && ember_.loaded() &&
           ember_.health().model_valid && ember_.health().finite_parameters;
}

bool ResilienceGateway::cloud_failure_is_retryable(GatewayFailure failure) noexcept {
    switch (failure) {
        case GatewayFailure::Network:
        case GatewayFailure::Timeout:
        case GatewayFailure::RateLimit:
        case GatewayFailure::ProviderFailure:
        case GatewayFailure::ModelUnavailable:
        case GatewayFailure::InvalidResponse:
            return true;
        case GatewayFailure::None:
        case GatewayFailure::LocalRuntimeFailure:
            return false;
    }
    return false;
}

GatewayResponse ResilienceGateway::try_cloud(std::string_view message) {
    GatewayResponse response;
    response.path = IntelligencePath::Cloud;

    if (!config_.cloud_enabled || !config_.allow_network) {
        response.failure = GatewayFailure::Network;
        response.error = "cloud path disabled by policy";
        return response;
    }
    if (!cloud_) {
        response.failure = GatewayFailure::ProviderFailure;
        response.error = "no cloud adapter configured";
        return response;
    }

    auto cloud_response = cloud_(message);
    response.success = cloud_response.success;
    response.text = std::move(cloud_response.text);
    response.failure = cloud_response.failure;
    response.provider = std::move(cloud_response.provider);
    response.model = std::move(cloud_response.model);
    response.error = std::move(cloud_response.error);
    return response;
}

GatewayResponse ResilienceGateway::try_ember(std::string_view message) {
    GatewayResponse response;
    response.path = IntelligencePath::Ember;
    response.ember_activated = true;

    if (!config_.ember_enabled) {
        response.failure = GatewayFailure::LocalRuntimeFailure;
        response.error = "EMBER disabled by policy";
        return response;
    }
    if (!ember_.loaded()) {
        response.failure = GatewayFailure::ModelUnavailable;
        response.error = "EMBER local model is not loaded";
        return response;
    }
    if (!ember_.health().model_valid || !ember_.health().finite_parameters) {
        response.failure = GatewayFailure::LocalRuntimeFailure;
        response.error = "EMBER local model failed integrity/health checks";
        return response;
    }

    try {
        auto local = ember_.chat(message);
        response.success = true;
        response.text = std::move(local.text);
        response.failure = GatewayFailure::None;
        response.model = ember_.identity().model_id;
        return response;
    } catch (const std::exception& ex) {
        response.failure = GatewayFailure::LocalRuntimeFailure;
        response.error = ex.what();
        return response;
    }
}

GatewayResponse ResilienceGateway::chat(const IntelligenceRequest& request) {
    const auto decision = IntelligenceRouter{} .decide(request);

    // Fresh-information requests must use the cloud path when network is
    // permitted. If that path fails, EMBER may continue the conversation but
    // must not pretend it has fresh information.
    if (decision.requires_network || request.require_fresh_information) {
        auto cloud = try_cloud(request.text);
        if (cloud.success) return cloud;

        if (config_.retry_cloud_once && cloud_failure_is_retryable(cloud.failure)) {
            auto retry = try_cloud(request.text);
            if (retry.success) return retry;
            cloud.error = retry.error.empty() ? cloud.error : retry.error;
            cloud.failure = retry.failure;
        }

        if (ember_available()) {
            auto local = try_ember(request.text);
            if (local.success) {
                local.error = "Cloud unavailable; EMBER answered locally. Fresh online information may be unavailable.";
                return local;
            }
            cloud.error = "cloud failed; EMBER fallback failed: " + local.error;
        }
        return cloud;
    }

    if (ember_available()) return try_ember(request.text);

    if (config_.cloud_enabled && config_.allow_network) return try_cloud(request.text);

    GatewayResponse unavailable;
    unavailable.path = IntelligencePath::Unavailable;
    unavailable.failure = GatewayFailure::ModelUnavailable;
    unavailable.error = "no available intelligence path";
    return unavailable;
}

} // namespace swarx
