#include "swarx/model_identity.hpp"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace swarx {
namespace {

class Fnv1a64 {
public:
    void bytes(const void* data, std::size_t size) noexcept {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash_ ^= static_cast<std::uint64_t>(p[i]);
            hash_ *= 1099511628211ULL;
        }
    }

    template <typename T>
    void value(const T& value) noexcept { bytes(&value, sizeof(T)); }

    std::string hex() const {
        std::ostringstream out;
        out << std::hex << std::setfill('0') << std::setw(16) << hash_;
        return out.str();
    }

private:
    std::uint64_t hash_ = 14695981039346656037ULL;
};

void add_tensor(Fnv1a64& h, const Tensor& tensor, std::size_t& count,
                std::size_t& bytes) {
    if (tensor.size() > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::overflow_error("SWARX model identity byte count overflow");
    count += tensor.size();
    bytes += tensor.size() * sizeof(float);
    h.value(tensor.size());
    if (!tensor.empty()) h.bytes(tensor.data(), tensor.size() * sizeof(float));
}

} // namespace

ModelIdentity ModelIdentityProvider::identify(const TransformerModel& model) {
    const auto& c = model.config();
    Fnv1a64 h;
    h.bytes("SWARX-MODEL-V1", 14);
    h.value(c.vocab_size);
    h.value(c.hidden_size);
    h.value(c.intermediate_size);
    h.value(c.num_heads);
    h.value(c.num_layers);
    h.value(c.max_sequence_length);
    h.value(c.rms_epsilon);

    std::size_t count = 0;
    std::size_t bytes = 0;
    add_tensor(h, model.token_embedding(), count, bytes);
    add_tensor(h, model.position_embedding(), count, bytes);
    add_tensor(h, model.final_norm_weight(), count, bytes);
    add_tensor(h, model.lm_head(), count, bytes);
    for (const auto& block : model.blocks()) {
        add_tensor(h, block.attention().query_weight(), count, bytes);
        add_tensor(h, block.attention().key_weight(), count, bytes);
        add_tensor(h, block.attention().value_weight(), count, bytes);
        add_tensor(h, block.attention().output_weight(), count, bytes);
        add_tensor(h, block.attention_norm_weight(), count, bytes);
        add_tensor(h, block.mlp_norm_weight(), count, bytes);
        add_tensor(h, block.up_weight(), count, bytes);
        add_tensor(h, block.down_weight(), count, bytes);
    }

    return {"fnv1a64", h.hex(), count, bytes};
}

std::string ModelIdentityProvider::fingerprint(const TransformerModel& model) {
    return identify(model).fingerprint;
}

} // namespace swarx
