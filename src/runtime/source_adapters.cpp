#include "swarx/source_adapters.hpp"

#include <utility>

namespace swarx {

OnlineSourceAdapter::OnlineSourceAdapter(
    std::string source_id,
    SourceCapability capability,
    OnlineIntent intent,
    std::shared_ptr<const OnlineInformationGateway> gateway)
    : source_id_(std::move(source_id)),
      capability_(capability),
      intent_(intent),
      gateway_(std::move(gateway)) {}

SourceResponse OnlineSourceAdapter::fetch(const SourceRequest& request) const {
    SourceResponse response;
    response.source_id = source_id_;
    response.capability = capability_;

    if (!gateway_) {
        response.error = "source gateway unavailable";
        return response;
    }

    OnlineRequest online_request;
    online_request.intent = intent_;
    online_request.query = request.query;
    online_request.location = request.location;
    online_request.latitude = request.latitude;
    online_request.longitude = request.longitude;
    online_request.timezone = request.timezone;
    online_request.source_policy = request.policy;

    const auto result = gateway_->fetch(online_request);
    response.success = result.success;
    response.url = result.source.url;
    response.status_code = result.source.status_code;
    response.content_type = result.source.content_type;
    response.body = result.source.body;
    response.evidence = result.evidence;
    response.error = result.error;
    return response;
}

SourceRegistry make_default_source_registry(const OnlineConfig& config) {
    SourceRegistry registry;
    auto gateway = std::make_shared<const OnlineInformationGateway>(config);

    registry.register_source(std::make_shared<const OnlineSourceAdapter>(
        "trendorafinds", SourceCapability::TrendoraFinds, OnlineIntent::Trendora, gateway));
    registry.register_source(std::make_shared<const OnlineSourceAdapter>(
        "timeapi", SourceCapability::Time, OnlineIntent::Time, gateway));
    registry.register_source(std::make_shared<const OnlineSourceAdapter>(
        "open-meteo", SourceCapability::Weather, OnlineIntent::Weather, gateway));
    registry.register_source(std::make_shared<const OnlineSourceAdapter>(
        "wikipedia", SourceCapability::GeneralWeb, OnlineIntent::General, gateway));

    return registry;
}

} // namespace swarx
