#pragma once

#include "swarx/inference.hpp"
#include "swarx/model_inspector.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace swarx {

struct PipelineConfig {
    InferenceConfig inference{};
    bool validate_model = true;
    bool collect_trace = true;
};

struct PipelineTrace {
    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t forward_passes = 0;
    std::size_t context_tokens = 0;
    bool model_valid = false;
    bool cancelled = false;
    bool stopped_on_eos = false;
    bool hit_context_limit = false;
    std::vector<std::string> stages;
};

struct PipelineResult {
    InferenceResult inference;
    ModelReport inspection;
    PipelineTrace trace;
};

// Stable SWARX execution boundary: input -> validation -> inference -> trace.
class EnginePipeline {
public:
    EnginePipeline();
    explicit EnginePipeline(TransformerModel model);
    explicit EnginePipeline(const std::string& weights_path);

    const InferenceSession& session() const noexcept { return session_; }
    InferenceSession& session() noexcept { return session_; }

    ModelReport inspect() const;
    PipelineResult run(std::string_view prompt,
                       PipelineConfig config = {},
                       TokenCallback on_token = {},
                       CancelCallback should_cancel = {});
    PipelineResult run(const std::vector<TokenId>& tokens,
                       PipelineConfig config = {},
                       TokenCallback on_token = {},
                       CancelCallback should_cancel = {});

    void reset() noexcept { session_.reset(); }

private:
    PipelineResult execute(const std::vector<TokenId>& tokens,
                           PipelineConfig config,
                           TokenCallback on_token,
                           CancelCallback should_cancel);

    InferenceSession session_;
};

} // namespace swarx
