#pragma once

#include "swarx/backend.hpp"
#include "swarx/inference.hpp"
#include "swarx/model_identity.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace swarx {

enum class ServiceState {
    Created,
    Ready,
    Failed,
    Stopped,
};

struct ServiceConfig {
    std::size_t max_prompt_tokens = 0;
    std::size_t max_batch_size = 16;
    bool expose_diagnostics = true;
};

struct ServiceHealth {
    ServiceState state = ServiceState::Created;
    std::string state_name;
    std::string backend;
    std::string model_fingerprint;
    std::size_t model_parameters = 0;
    std::size_t requests_served = 0;
    std::size_t failed_requests = 0;
};

struct CompletionRequest {
    std::string prompt;
    InferenceConfig inference{};
};

struct CompletionResponse {
    std::uint64_t request_id = 0;
    std::string text;
    std::vector<TokenId> tokens;
    RuntimeStats stats{};
    std::string model_fingerprint;
};

struct BatchCompletionResponse {
    std::uint64_t request_id = 0;
    std::vector<CompletionResponse> results;
    double elapsed_ms = 0.0;
};

// Stable serving boundary for SWARX products. Product layers can depend on
// this interface without knowing how tensors, models or kernels are executed.
class ServingRuntime {
public:
    explicit ServingRuntime(ServiceConfig config = {});
    explicit ServingRuntime(TransformerModel model, ServiceConfig config = {});

    ServiceHealth health() const;
    void load_weights(const std::string& path);
    void reset();
    void stop() noexcept;

    CompletionResponse complete(const CompletionRequest& request);
    BatchCompletionResponse complete_batch(const std::vector<CompletionRequest>& requests);

    const ServiceConfig& config() const noexcept { return config_; }
    const TransformerModel& model() const noexcept { return session_.model(); }
    const ComputeBackend& backend() const noexcept { return BackendRegistry::default_backend(); }

private:
    void ensure_ready_locked() const;
    static std::string state_name(ServiceState state);

    mutable std::mutex mutex_;
    ServiceConfig config_;
    InferenceSession session_;
    ServiceState state_ = ServiceState::Created;
    std::uint64_t next_request_id_ = 1;
    std::size_t requests_served_ = 0;
    std::size_t failed_requests_ = 0;
};

} // namespace swarx
