#pragma once

#include "swarx/online.hpp"
#include "swarx/source_registry.hpp"

#include <memory>

namespace swarx {

// Adapts a controlled OnlineInformationGateway into the generic Source contract.
// This keeps provider-specific URLs and transport details below the intelligence layer.
class OnlineSourceAdapter final : public Source {
public:
    OnlineSourceAdapter(std::string source_id,
                        SourceCapability capability,
                        OnlineIntent intent,
                        std::shared_ptr<const OnlineInformationGateway> gateway);

    const std::string& id() const noexcept override { return source_id_; }
    SourceCapability capability() const noexcept override { return capability_; }
    SourceResponse fetch(const SourceRequest& request) const override;

private:
    std::string source_id_;
    SourceCapability capability_;
    OnlineIntent intent_;
    std::shared_ptr<const OnlineInformationGateway> gateway_;
};

// Builds the production source set used by the KnowledgeFabric.
// TrendoraFinds is the owned knowledge hub; other sources provide live or
// general-purpose information when their capabilities are requested.
SourceRegistry make_default_source_registry(const OnlineConfig& config = {});

} // namespace swarx
