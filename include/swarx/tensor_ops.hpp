#pragma once

#include "swarx/tensor.hpp"

namespace swarx {

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor transpose2d(const Tensor& input);
Tensor softmax(const Tensor& input);
Tensor rms_norm(const Tensor& input, const Tensor& weight, float epsilon = 1e-5f);
Tensor add(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);

} // namespace swarx
