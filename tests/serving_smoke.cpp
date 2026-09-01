#include "swarx/serving.hpp"

#include <cassert>
#include <string>
#include <vector>

int main() {
    swarx::ServiceConfig config;
    config.max_batch_size = 4;
    config.max_prompt_tokens = 64;

    swarx::ServingRuntime runtime(config);
    const auto initial = runtime.health();
    assert(initial.state == swarx::ServiceState::Ready);
    assert(initial.state_name == "ready");
    assert(initial.backend == "cpu");
    assert(!initial.model_fingerprint.empty());

    swarx::CompletionRequest request;
    request.prompt = "hello swarx";
    request.inference.generation.max_new_tokens = 2;
    request.inference.generation.temperature = 0.0f;

    const auto completion = runtime.complete(request);
    assert(completion.request_id != 0);
    assert(completion.model_fingerprint == initial.model_fingerprint);
    assert(completion.stats.prompt_tokens > 0);

    std::vector<swarx::CompletionRequest> batch(2, request);
    batch[1].prompt = "second request";
    const auto batched = runtime.complete_batch(batch);
    assert(batched.request_id != 0);
    assert(batched.results.size() == 2);
    assert(batched.results[0].request_id != batched.results[1].request_id);

    const auto after = runtime.health();
    assert(after.requests_served == 3);
    assert(after.failed_requests == 0);

    runtime.stop();
    assert(runtime.health().state == swarx::ServiceState::Stopped);

    return 0;
}
