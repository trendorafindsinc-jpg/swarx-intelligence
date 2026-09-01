#include "swarx/engine.hpp"
#include "swarx/generation.hpp"
#include "swarx/inference.hpp"
#include "swarx/kv_cache.hpp"
#include "swarx/model.hpp"
#include "swarx/model_identity.hpp"
#include "swarx/model_inspector.hpp"
#include "swarx/sampler.hpp"
#include "swarx/tensor.hpp"
#include "swarx/transformer.hpp"
#include "swarx/weights.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    swarx::Tensor tensor({2, 3});
    assert(tensor.size() == 6);

    swarx::Sampler sampler(1234);
    assert(sampler.greedy({-1.0f, 3.0f, 2.0f}) == 1);
    assert(sampler.sample({0.0f, 100.0f, 1.0f}, swarx::SamplingConfig{0.7f, 2, 0.9f}) == 1);

    swarx::Engine engine;
    const auto tokens = engine.tokenize("hello world");
    assert(tokens.size() == 2);
    assert(engine.generate("hello").find("SWARX Intelligence runtime is ready") != std::string::npos);

    swarx::TransformerConfig block_config;
    block_config.hidden_size = 8;
    block_config.intermediate_size = 16;
    block_config.num_heads = 2;
    swarx::TransformerBlock block(block_config);

    swarx::Tensor sequence({3, 8});
    for (std::size_t i = 0; i < sequence.size(); ++i)
        sequence[i] = static_cast<float>(i + 1) * 0.05f;

    const swarx::Tensor output = block.forward(sequence);
    assert(output.shape().size() == 2);
    assert(output.shape()[0] == 3);
    assert(output.shape()[1] == 8);
    for (std::size_t i = 0; i < output.size(); ++i) assert(std::isfinite(output[i]));

    swarx::ModelConfig model_config;
    model_config.vocab_size = 64;
    model_config.hidden_size = 16;
    model_config.intermediate_size = 32;
    model_config.num_heads = 4;
    model_config.num_layers = 3;
    model_config.max_sequence_length = 32;

    swarx::TransformerModel model(model_config);
    const auto report = swarx::ModelInspector::inspect(model);
    assert(report.parameter_count > 0);
    assert(report.parameter_bytes == report.parameter_count * sizeof(float));
    assert(report.fingerprint != 0);
    assert(report.finite);
    assert(report.warnings.empty());
    swarx::ModelInspector::validate(model);

    const auto identity = swarx::ModelIdentityProvider::identify(model);
    assert(identity.algorithm == "fnv1a64");
    assert(identity.fingerprint.size() == 16);
    assert(identity.parameter_count == report.parameter_count);
    assert(identity.parameter_bytes == report.parameter_bytes);
    assert(identity.fingerprint == swarx::ModelIdentityProvider::fingerprint(model));

    const std::vector<std::uint32_t> prompt_tokens = {1, 7, 12, 4};
    const swarx::Tensor hidden = model.forward(prompt_tokens);
    assert(hidden.shape()[0] == prompt_tokens.size());
    assert(hidden.shape()[1] == model_config.hidden_size);
    for (std::size_t i = 0; i < hidden.size(); ++i) assert(std::isfinite(hidden[i]));

    const swarx::Tensor next_logits = model.next_token_logits(prompt_tokens);
    assert(next_logits.shape()[0] == model_config.vocab_size);
    for (std::size_t i = 0; i < next_logits.size(); ++i) assert(std::isfinite(next_logits[i]));

    const std::size_t greedy_token = sampler.greedy(
        std::vector<float>(next_logits.data(), next_logits.data() + next_logits.size()));
    assert(greedy_token < model_config.vocab_size);

    swarx::GenerationConfig generation_config;
    generation_config.max_new_tokens = 6;
    generation_config.temperature = 0.0f;
    generation_config.repetition_penalty = 1.1f;
    generation_config.slide_context = true;

    swarx::Generator generator(model, 1234);
    std::size_t streamed = 0;
    const auto generated = generator.generate(
        prompt_tokens, generation_config,
        [&](std::uint32_t token) { ++streamed; assert(token < model_config.vocab_size); });
    assert(generated.tokens.size() == prompt_tokens.size() + 6);
    assert(generated.generated_tokens == 6);
    assert(streamed == 6);
    assert(!generated.stopped_on_eos);
    assert(!generated.cancelled);

    swarx::GenerationConfig advanced = generation_config;
    advanced.temperature = 0.8f;
    advanced.top_k = 8;
    advanced.top_p = 0.9f;
    const auto advanced_generated = generator.generate(prompt_tokens, advanced);
    assert(advanced_generated.generated_tokens == 6);

    std::size_t cancel_checks = 0;
    const auto cancelled = generator.generate(
        prompt_tokens, generation_config, {}, [&]() { return ++cancel_checks >= 2; });
    assert(cancelled.cancelled);
    assert(cancelled.generated_tokens == 1);

    swarx::GenerationConfig sliding;
    sliding.max_new_tokens = 5;
    sliding.temperature = 0.0f;
    sliding.slide_context = true;
    const auto long_generated = generator.generate(
        std::vector<std::uint32_t>(model_config.max_sequence_length, 3), sliding);
    assert(long_generated.generated_tokens == 5);
    assert(!long_generated.hit_context_limit);

    swarx::KVCache cache(model_config.num_layers, model_config.max_sequence_length,
                         model_config.hidden_size);
    swarx::Tensor key({model_config.hidden_size});
    swarx::Tensor value({model_config.hidden_size});
    for (std::size_t i = 0; i < key.size(); ++i) { key[i] = 1.0f; value[i] = 2.0f; }
    for (std::size_t layer = 0; layer < model_config.num_layers; ++layer)
        cache.append(layer, key, value);
    assert(cache.size() == 1);
    assert(cache.keys(0)[0] == 1.0f);
    assert(cache.values(0)[0] == 2.0f);
    cache.reset();
    assert(cache.empty());

    const std::string weights_path = "swarx-test-model.swarx";
    swarx::WeightsWriter::write(weights_path, model);
    const auto info = swarx::WeightsReader::inspect(weights_path);
    assert(info.tensor_count == 4 + 8 * model_config.num_layers);
    assert(info.config.vocab_size == model_config.vocab_size);
    assert(info.config.hidden_size == model_config.hidden_size);
    assert(info.config.intermediate_size == model_config.intermediate_size);
    assert(info.config.num_heads == model_config.num_heads);
    assert(info.config.num_layers == model_config.num_layers);
    assert(info.config.max_sequence_length == model_config.max_sequence_length);

    swarx::TransformerModel loaded = swarx::WeightsReader::load(weights_path);
    swarx::ModelInspector::validate(loaded);
    assert(loaded.config().vocab_size == model.config().vocab_size);
    const auto loaded_report = swarx::ModelInspector::inspect(loaded);
    assert(loaded_report.fingerprint == report.fingerprint);
    assert(swarx::ModelIdentityProvider::fingerprint(loaded) == identity.fingerprint);
    const auto original_logits = model.next_token_logits(prompt_tokens);
    const auto loaded_logits = loaded.next_token_logits(prompt_tokens);
    assert(original_logits.shape() == loaded_logits.shape());
    for (std::size_t i = 0; i < original_logits.size(); ++i)
        assert(std::fabs(original_logits[i] - loaded_logits[i]) < 1e-6f);

    swarx::Generator loaded_generator(loaded, 1234);
    const auto loaded_generated = loaded_generator.generate(prompt_tokens, generation_config);
    assert(loaded_generated.tokens == generated.tokens);

    swarx::InferenceSession session(model);
    assert(session.cache().layers() == model_config.num_layers);
    assert(session.cache().capacity() == model_config.max_sequence_length);
    assert(session.model_config().vocab_size == model_config.vocab_size);
    const auto session_result = session.generate(prompt_tokens, {generation_config, false});
    assert(session_result.stats.generated_tokens == generation_config.max_new_tokens);
    assert(session_result.tokens.size() == prompt_tokens.size() + generation_config.max_new_tokens);
    assert(session.last_stats().total_tokens == session_result.tokens.size());

    const auto batch = session.generate_batch(
        std::vector<std::string>{"alpha", "beta", "gamma"},
        {generation_config, false});
    assert(batch.results.size() == 3);
    assert(batch.total_generated_tokens == 3 * generation_config.max_new_tokens);
    assert(batch.total_prompt_tokens > 0);
    assert(batch.elapsed_ms >= 0.0);
    assert(batch.generated_tokens_per_second >= 0.0);

    const auto token_batch = session.generate_batch(
        std::vector<std::vector<std::uint32_t>>{{1, 2}, {3, 4, 5}},
        {generation_config, false});
    assert(token_batch.results.size() == 2);
    assert(token_batch.total_generated_tokens == 2 * generation_config.max_new_tokens);

    session.save_weights("swarx-session-test.swarx");
    swarx::InferenceSession restored("swarx-session-test.swarx");
    assert(restored.model_config().hidden_size == model_config.hidden_size);
    assert(restored.cache().capacity() == model_config.max_sequence_length);
    std::remove("swarx-session-test.swarx");

    std::remove(weights_path.c_str());
    return 0;
}
