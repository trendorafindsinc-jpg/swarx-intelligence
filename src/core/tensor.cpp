#include "swarx/tensor.hpp"

#include <numeric>
#include <stdexcept>

namespace swarx {

Tensor::Tensor(std::vector<std::size_t> shape) : shape_(std::move(shape)) {
    const auto elements = std::accumulate(
        shape_.begin(), shape_.end(), std::size_t{1},
        [](std::size_t a, std::size_t b) { return a * b; });
    data_.resize(elements);
}

} // namespace swarx
