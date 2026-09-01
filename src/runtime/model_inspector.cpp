#include "swarx/model_inspector.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace swarx {
namespace {

class Fingerprint {
public:
    Fingerprint() = default;

    void update_bytes(const void* data, std::size_t size) noexcept {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash_ ^= bytes[i];
            hash_ *= kPrime;
        }
    }

    template <typename T>
    void update(const T& value) noexcept {
        update_bytes(&value, sizeof(T));
    }

    void update_tensor(const Tensor& tensor) noexcept {
        const auto& shape = tensor.shape();
        update(shape.size());
        for (const auto dimension : shape) update(dimension);
        update_bytes(tensor.data(), tensor.size() * sizeof(float));
    }

    std::uint64_t value() const noexcept { return hash_; }

private:
    static constexpr std::uint64_t kOffset = 14695981039346656037ull;
    static constexpr std::uint64_t kPrime = 1099511628211ull;
    std::uint64_t hash_ = kOffset;
};

void inspect_tensor(const Tensor& tensor, const char* name, ModelReport& report,
                    Fingerprint& fingerprint) {
    if (tensor.size() > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::overflow_error("SWARX parameter byte count overflow");
    report.parameter_count += tensor.size();
    report.parameter_bytes += tensor.size() * sizeof(float);
    fingerprint.update_tensor(tensor);
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        if (!std::isfinite(tensor[i])) {
            report.finite = false;
            report.warnings.emplace_back(std::string("non-finite value in ") + name);
            return;
        }
    }
}

} // namespace

ModelReport ModelInspector::inspect(const TransformerModel& model) {
    const auto& c = model.config();
    ModelReport report;
    Fingerprint fingerprint;
    fingerprint.update(c.vocab_size);
    fingerprint.update(c.hidden_size);
    fingerprint.update(c.intermediate_size);
    fingerprint.update(c.num_heads);
    fingerprint.update(c.num_layers);
    fingerprint.update(c.max_sequence_length);
    fingerprint.update(c.rms_epsilon);

    report.vocab_size = c.vocab_size;
    report.hidden_size = c.hidden_size;
    report.intermediate_size = c.intermediate_size;
    report.num_heads = c.num_heads;
    report.num_layers = c.num_layers;
    report.max_sequence_length = c.max_sequence_length;

    if (c.vocab_size == 0) report.warnings.emplace_back("vocab_size is zero");
    if (c.hidden_size == 0) report.warnings.emplace_back("hidden_size is zero");
    if (c.intermediate_size == 0) report.warnings.emplace_back("intermediate_size is zero");
    if (c.num_layers == 0) report.warnings.emplace_back("num_layers is zero");
    if (c.num_heads == 0) report.warnings.emplace_back("num_heads is zero");
    if (c.num_heads != 0 && c.hidden_size % c.num_heads != 0)
        report.warnings.emplace_back("hidden_size is not divisible by num_heads");
    if (c.max_sequence_length == 0)
        report.warnings.emplace_back("max_sequence_length is zero");
    if (!(c.rms_epsilon > 0.0f) || !std::isfinite(c.rms_epsilon))
        report.warnings.emplace_back("rms_epsilon must be finite and positive");

    inspect_tensor(model.token_embedding(), "token_embedding", report, fingerprint);
    inspect_tensor(model.position_embedding(), "position_embedding", report, fingerprint);
    inspect_tensor(model.final_norm_weight(), "final_norm_weight", report, fingerprint);
    inspect_tensor(model.lm_head(), "lm_head", report, fingerprint);

    if (model.blocks().size() != c.num_layers)
        report.warnings.emplace_back("block count does not match num_layers");

    for (std::size_t i = 0; i < model.blocks().size(); ++i) {
        const auto& block = model.blocks()[i];
        inspect_tensor(block.attention().query_weight(), "attention.query_weight", report, fingerprint);
        inspect_tensor(block.attention().key_weight(), "attention.key_weight", report, fingerprint);
        inspect_tensor(block.attention().value_weight(), "attention.value_weight", report, fingerprint);
        inspect_tensor(block.attention().output_weight(), "attention.output_weight", report, fingerprint);
        inspect_tensor(block.attention_norm_weight(), "attention_norm_weight", report, fingerprint);
        inspect_tensor(block.mlp_norm_weight(), "mlp_norm_weight", report, fingerprint);
        inspect_tensor(block.up_weight(), "up_weight", report, fingerprint);
        inspect_tensor(block.down_weight(), "down_weight", report, fingerprint);
    }
    report.fingerprint = fingerprint.value();
    return report;
}

void ModelInspector::validate(const TransformerModel& model) {
    const auto report = inspect(model);
    if (!report.warnings.empty())
        throw std::invalid_argument("invalid SWARX model: " + report.warnings.front());
    if (!report.finite)
        throw std::invalid_argument("invalid SWARX model: non-finite parameter value");
}

} // namespace swarx
