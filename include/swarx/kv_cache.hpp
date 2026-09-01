#pragma once

#include "swarx/tensor.hpp"

#include <cstddef>
#include <vector>

namespace swarx {

// Runtime-owned key/value storage for incremental transformer execution.
// The current reference model can use it as a validated cache boundary;
// block-level cache consumption can be added without changing the public API.
class KVCache {
public:
    KVCache() = default;
    KVCache(std::size_t layers, std::size_t max_sequence, std::size_t hidden);

    void reset() noexcept;
    void append(std::size_t layer, const Tensor& key, const Tensor& value);

    std::size_t layers() const noexcept { return layers_; }
    std::size_t size() const noexcept { return sequence_length_; }
    std::size_t capacity() const noexcept { return max_sequence_; }
    bool empty() const noexcept { return sequence_length_ == 0; }

    const Tensor& keys(std::size_t layer) const;
    const Tensor& values(std::size_t layer) const;

private:
    std::size_t layers_ = 0;
    std::size_t max_sequence_ = 0;
    std::size_t hidden_ = 0;
    std::size_t sequence_length_ = 0;
    std::vector<Tensor> keys_;
    std::vector<Tensor> values_;
};

} // namespace swarx
