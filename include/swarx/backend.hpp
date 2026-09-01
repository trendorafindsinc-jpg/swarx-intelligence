#pragma once

#include "swarx/tensor.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace swarx {

enum class BackendOp {
    MatMul,
    RMSNorm,
    Softmax,
    GELU,
};

struct BackendCapabilities {
    bool cpu = false;
    bool mobile = false;
    bool accelerator = false;
    bool int8 = false;
    bool fp16 = false;
    bool bf16 = false;
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    virtual std::string_view name() const noexcept = 0;
    virtual BackendCapabilities capabilities() const noexcept = 0;
    virtual bool supports(BackendOp op) const noexcept = 0;

    virtual Tensor matmul(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor rms_norm(const Tensor& input, const Tensor& weight,
                            float epsilon) const = 0;
    virtual Tensor softmax(const Tensor& input) const = 0;
    virtual Tensor gelu(const Tensor& input) const = 0;
};

class CpuBackend final : public ComputeBackend {
public:
    std::string_view name() const noexcept override { return "cpu"; }
    BackendCapabilities capabilities() const noexcept override;
    bool supports(BackendOp op) const noexcept override;

    Tensor matmul(const Tensor& a, const Tensor& b) const override;
    Tensor rms_norm(const Tensor& input, const Tensor& weight,
                    float epsilon) const override;
    Tensor softmax(const Tensor& input) const override;
    Tensor gelu(const Tensor& input) const override;
};

class BackendRegistry {
public:
    static const ComputeBackend& cpu() noexcept;
    static const ComputeBackend& default_backend() noexcept;
};

} // namespace swarx
