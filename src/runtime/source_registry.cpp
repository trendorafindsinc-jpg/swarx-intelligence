#include "swarx/source_registry.hpp"

#include <algorithm>

namespace swarx {

bool SourceRegistry::register_source(SourcePtr source) {
    if (!source || source->id().empty()) return false;
    if (find(source->id())) return false;
    sources_.push_back(std::move(source));
    return true;
}

bool SourceRegistry::unregister_source(const std::string& id) {
    const auto old_size = sources_.size();
    sources_.erase(std::remove_if(sources_.begin(), sources_.end(),
        [&](const SourcePtr& source) { return source && source->id() == id; }), sources_.end());
    return sources_.size() != old_size;
}

SourcePtr SourceRegistry::find(SourceCapability capability) const {
    for (const auto& source : sources_)
        if (source && source->capability() == capability) return source;
    return {};
}

SourcePtr SourceRegistry::find(const std::string& id) const {
    for (const auto& source : sources_)
        if (source && source->id() == id) return source;
    return {};
}

} // namespace swarx
