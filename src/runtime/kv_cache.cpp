#include "swarx/kv_cache.hpp"

#include <stdexcept>

namespace swarx {

KVCache::KVCache(std::size_t layers, std::size_t max_sequence, std::size_t hidden)
    : layers_(layers), max_sequence_(max_sequence), hidden_(hidden),
      keys_(layers, Tensor({max_sequence, hidden})),
      values_(layers, Tensor({max_sequence, hidden})) {
    if (layers == 0 || max_sequence == 0 || hidden == 0)
        throw std::invalid_argument("KV cache dimensions must be non-zero");
}

void KVCache::reset() noexcept {
    sequence_length_ = 0;
}

void KVCache::append(std::size_t layer, const Tensor& key, const Tensor& value) {
    if (layer >= layers_) throw std::out_of_range("KV cache layer out of range");
    if (key.shape() != std::vector<std::size_t>{hidden_} ||
        value.shape() != std::vector<std::size_t>{hidden_})
        throw std::invalid_argument("KV cache entries must have shape [hidden]");
    if (sequence_length_ >= max_sequence_)
        throw std::length_error("KV cache context capacity exceeded");

    const std::size_t offset = sequence_length_ * hidden_;
    for (std::size_t i = 0; i < hidden_; ++i) {
        keys_[layer][offset + i] = key[i];
        values_[layer][offset + i] = value[i];
    }
    if (layer + 1 == layers_) ++sequence_length_;
}

const Tensor& KVCache::keys(std::size_t layer) const {
    if (layer >= layers_) throw std::out_of_range("KV cache layer out of range");
    return keys_[layer];
}

const Tensor& KVCache::values(std::size_t layer) const {
    if (layer >= layers_) throw std::out_of_range("KV cache layer out of range");
    return values_[layer];
}

} // namespace swarx
