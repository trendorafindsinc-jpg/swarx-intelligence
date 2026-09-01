#include "swarx/engine_pipeline.hpp"

#include <chrono>
#include <stdexcept>

namespace swarx {

EnginePipeline::EnginePipeline() = default;
EnginePipeline::EnginePipeline(TransformerModel model) : session_(std::move(model)) {}
EnginePipeline::EnginePipeline(const std::string& weights_path) : session_(weights_path) {}

ModelInspection EnginePipeline::inspect() const {
    return inspect_model(session_.model());
}

PipelineResult EnginePipeline::run(std::string_view prompt,
                                   PipelineConfig config,
                                   TokenCallback on_token,
                                   CancelCallback should_cancel) {
    const auto tokens = session_.model_config().vocab_size == 0
        ? std::vector<TokenId>{}
        : session_.generate(prompt, config.inference).tokens;
    if (tokens.empty() && !prompt.empty()) {
        // Fall back to the regular token path so callbacks/stats are preserved.
        const auto prompt_tokens = session_.model_config().vocab_size == 0
            ? std::vector<TokenId>{}
            : Tokenizer{}.encode(prompt);
        return execute(prompt_tokens, config, std::move(on_token), std::move(should_cancel));
    }
    return execute(tokens, config, std::move(on_token), std::move(should_cancel));
}

PipelineResult EnginePipeline::run(const std::vector<TokenId>& tokens,
                                   PipelineConfig config,
                                   TokenCallback on_token,
                                   CancelCallback should_cancel) {
    return execute(tokens, config, std::move(on_token), std::move(should_cancel));
}

PipelineResult EnginePipeline::execute(const std::vector<TokenId>& tokens,
                                       PipelineConfig config,
                                       TokenCallback on_token,
                                       CancelCallback should_cancel) {
    PipelineResult result;
    result.trace.prompt_tokens = tokens.size();
    result.trace.context_tokens = tokens.size();
    result.trace.stages.push_back("input");

    if (config.validate_model) {
        result.inspection = inspect();
        result.trace.model_valid = result.inspection.valid;
        result.trace.stages.push_back("validation");
        if (!result.inspection.valid)
            throw std::runtime_error("SWARX model validation failed");
    } else {
        result.trace.model_valid = true;
    }

    result.trace.stages.push_back("tokenized");
    result.trace.stages.push_back("inference");
    result.inference = session_.generate(tokens, config.inference,
                                         std::move(on_token),
                                         std::move(should_cancel));
    result.trace.generated_tokens = result.inference.stats.generated_tokens;
    result.trace.context_tokens = result.inference.tokens.size();
    result.trace.forward_passes = result.inference.stats.total_tokens;
    result.trace.cancelled = result.inference.stats.cancelled;
    result.trace.stopped_on_eos = result.inference.stats.stopped_on_eos;
    result.trace.stages.push_back("sampling");
    result.trace.stages.push_back("complete");
    return result;
}

} // namespace swarx
