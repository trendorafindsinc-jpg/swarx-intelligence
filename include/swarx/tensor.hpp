#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace swarx {

enum class DType : std::uint8_t {
    F32,
};

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape);

    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    std::size_t size() const noexcept { return data_.size(); }
    float* data() noexcept { return data_.data(); }
    const float* data() const noexcept { return data_.data(); }

    float& operator[](std::size_t i) noexcept { return data_[i]; }
    const float& operator[](std::size_t i) const noexcept { return data_[i]; }

private:
    std::vector<std::size_t> shape_;
    std::vector<float> data_;
};

} // namespace swarx
