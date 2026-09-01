#pragma once

#include "swarx/model.hpp"

#include <cstddef>
#include <string>

namespace swarx {

struct ModelIdentity {
    std::string algorithm = "fnv1a64";
    std::string fingerprint;
    std::size_t parameter_count = 0;
    std::size_t parameter_bytes = 0;
};

// Produces a deterministic identity from model configuration and parameter
// bytes. This is intentionally dependency-free so SWARX artifacts can be
// identified consistently on mobile, desktop and server runtimes.
class ModelIdentityProvider {
public:
    static ModelIdentity identify(const TransformerModel& model);
    static std::string fingerprint(const TransformerModel& model);
};

} // namespace swarx
