#include "swarx/serving.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace swarx {

ServingRuntime::ServingRuntime(ServiceConfig config)
    : config_(config), session_() {
    state_ = ServiceState::Ready;
}

ServingRuntime::ServingRuntime(TransformerModel model, ServiceConfig config)
    : config_(config), session_(std::move(model)) {
    state_ = ServiceState::Ready;
}

std::string ServingRuntime::state_name(ServiceState state) {
    switch (state) {
        case ServiceState::Created: return "created";
        case ServiceState::Ready: return "ready";
        case ServiceState::Failed: return "failed";
        case ServiceState::Stopped: return "stopped";
    }
    return "unknown";
}

void ServingRuntime::ensure_ready_locked() const {
    if (state_ != ServiceState::Ready) {
        throw std::runtime_error("SWARX serving runtime is not ready");
    }
}

ServiceHealth ServingRuntime::health() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto identity = ModelIdentityProvider::identify(session_.model());
    ServiceHealth result;
    result.state = state_;
    result.state_name = state_name(state_);
    result.backend = std::string(backend().name());
    result.model_fingerprint = identity.fingerprint;
    result.model_parameters = identity.parameter_count;
    result.requests_served = requests_served_;
    result.failed_requests = failed_requests_;
    return result;
}

void ServingRuntime::load_weights(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        session_.load_weights(path);
        state_ = ServiceState::Ready;
    } catch (...) {
        state_ = ServiceState::Failed;
        ++failed_requests_;
        throw;
    }
}

void ServingRuntime::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == ServiceState::Stopped) {
        throw std::runtime_error("cannot reset a stopped SWARX runtime");
    }
    session_.reset();
    state_ = ServiceState::Ready;
}

void ServingRuntime::stop() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = ServiceState::Stopped;
}

CompletionResponse ServingRuntime::complete(const CompletionRequest& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_ready_locked();
    if (request.prompt.empty()) {
        ++failed_requests_;
        throw std::invalid_argument("completion prompt cannot be empty");
    }

    try {
        const auto result = session_.generate(request.prompt, request.inference);
        if (config_.max_prompt_tokens != 0 &&
            result.stats.prompt_tokens > config_.max_prompt_tokens) {
            throw std::length_error("completion prompt exceeds SWARX prompt limit");
        }

        CompletionResponse response;
        response.request_id = next_request_id_++;
        response.text = result.text;
        response.tokens = result.tokens;
        response.stats = result.stats;
        response.model_fingerprint = ModelIdentityProvider::fingerprint(session_.model());
        ++requests_served_;
        return response;
    } catch (...) {
        ++failed_requests_;
        throw;
    }
}

BatchCompletionResponse ServingRuntime::complete_batch(
    const std::vector<CompletionRequest>& requests) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_ready_locked();
    if (requests.empty()) {
        throw std::invalid_argument("completion batch cannot be empty");
    }
    if (requests.size() > config_.max_batch_size) {
        throw std::length_error("completion batch exceeds SWARX batch limit");
    }

    const auto started = std::chrono::steady_clock::now();
    BatchCompletionResponse response;
    response.request_id = next_request_id_++;
    response.results.reserve(requests.size());

    try {
        std::vector<std::string> prompts;
        prompts.reserve(requests.size());
        for (const auto& request : requests) {
            if (request.prompt.empty()) {
                throw std::invalid_argument("completion prompt cannot be empty");
            }
            prompts.push_back(request.prompt);
        }

        const auto batch = session_.generate_batch(prompts, requests.front().inference);
        const auto fingerprint = ModelIdentityProvider::fingerprint(session_.model());
        for (auto& result : batch.results) {
            CompletionResponse item;
            item.request_id = next_request_id_++;
            item.text = std::move(result.text);
            item.tokens = std::move(result.tokens);
            item.stats = result.stats;
            item.model_fingerprint = fingerprint;
            response.results.push_back(std::move(item));
            ++requests_served_;
        }
        response.elapsed_ms = batch.elapsed_ms;
        return response;
    } catch (...) {
        ++failed_requests_;
        throw;
    }
}

} // namespace swarx
