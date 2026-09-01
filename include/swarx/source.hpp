#pragma once

#include "swarx/source_response.hpp"
#include <memory>
#include <string>

namespace swarx {

class Source {
public:
    virtual ~Source() = default;
    virtual const std::string& id() const noexcept = 0;
    virtual SourceCapability capability() const noexcept = 0;
    virtual SourceResponse fetch(const SourceRequest& request) const = 0;
};

using SourcePtr = std::shared_ptr<const Source>;

} // namespace swarx
