#pragma once

#include "swarx/source.hpp"
#include <memory>
#include <vector>

namespace swarx {

class SourceRegistry {
public:
    bool register_source(SourcePtr source);
    bool unregister_source(const std::string& id);
    SourcePtr find(SourceCapability capability) const;
    SourcePtr find(const std::string& id) const;
    const std::vector<SourcePtr>& sources() const noexcept { return sources_; }

private:
    std::vector<SourcePtr> sources_;
};

} // namespace swarx
