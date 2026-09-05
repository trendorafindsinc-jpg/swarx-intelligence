#pragma once

#include "swarx/intelligence_router.hpp"
#include "swarx/runtime.hpp"

#include <functional>
#include <string>
#include <string_view>

namespace swarx {

enum class IntelligencePath {
    Cloud,
    Ember,
    Unavailable,
};

enum class GatewayFailure {
    None,
    Network,
    Timeout,
    RateLimit,
    ProviderFailure,
    ModelUnavailable,
    InvalidResponse,
    LocalRuntimeFailure,
};

struct CloudResponse {
    bool success = false;
    std::string text;
    GatewayFailure failure = GatewayFailure::None;
    std::string provider;
    std::string model;
    std::string error;
};

using CloudAdapter = std::function<CloudResponse(std::string_view)>;

struct ResilienceGatewayConfig {
    bool cloud_enabled = true;
    bool ember_enabled = true;
    bool allow_network = true;
    bool retry_cloud_once = false;
    RuntimeConfig ember_runtime{};
};

struct GatewayResponse {
    bool success = false;
    IntelligencePath path = IntelligencePath::Unavailable;
    GatewayFailure failure = GatewayFailure::None;
    std::string text;
    std::string provider;
    std::string model;
    std::string error;
    bool ember_activated = false;
};

// EMBER resilience boundary. Cloud providers are injected as replaceable
// adapters; the local SWARX runtime is the persistent fallback path.
class ResilienceGateway {
public:
    explicit ResilienceGateway(ResilienceGatewayConfig config = {});
    ResilienceGateway(ResilienceGatewayConfig config, CloudAdapter cloud);

    void set_cloud_adapter(CloudAdapter cloud);
    void set_ember_runtime(Runtime runtime);
    void load_ember(const std::string& weights_path);

    GatewayResponse chat(const IntelligenceRequest& request);

    bool ember_available() const noexcept;
    bool cloud_enabled() const noexcept { return config_.cloud_enabled; }
    bool network_allowed() const noexcept { return config_.allow_network; }

private:
    GatewayResponse try_cloud(std::string_view message);
    GatewayResponse try_ember(std::string_view message);
    static bool cloud_failure_is_retryable(GatewayFailure failure) noexcept;

    ResilienceGatewayConfig config_{};
    CloudAdapter cloud_{};
    Runtime ember_{};
};

} // namespace swarx
