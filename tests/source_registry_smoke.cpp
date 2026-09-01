#include "swarx/source_registry.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

namespace {
class FakeSource final : public swarx::Source {
public:
    FakeSource(std::string name, swarx::SourceCapability cap)
        : name_(std::move(name)), cap_(cap) {}

    const std::string& id() const noexcept override { return name_; }
    swarx::SourceCapability capability() const noexcept override { return cap_; }

    swarx::SourceResponse fetch(const swarx::SourceRequest&) const override {
        swarx::SourceResponse response;
        response.success = true;
        response.source_id = name_;
        response.capability = cap_;
        response.body = "fake evidence";
        return response;
    }

private:
    std::string name_;
    swarx::SourceCapability cap_;
};

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "source-registry test failure: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}
} // namespace

int main() {
    swarx::SourceRegistry registry;

    require(!registry.register_source(nullptr), "null source must be rejected");

    auto weather = std::make_shared<FakeSource>(
        "weather-primary", swarx::SourceCapability::Weather);
    require(registry.register_source(weather), "first source registration must succeed");
    require(!registry.register_source(weather), "same source pointer must not be registered twice");
    require(!registry.register_source(std::make_shared<FakeSource>(
                "weather-primary", swarx::SourceCapability::Weather)),
            "duplicate source id must be rejected");

    const auto by_id = registry.find("weather-primary");
    require(by_id != nullptr, "source lookup by id must succeed");
    require(by_id->id() == "weather-primary", "source id must be preserved");

    const auto by_capability = registry.find(swarx::SourceCapability::Weather);
    require(by_capability != nullptr, "source lookup by capability must succeed");
    require(by_capability->id() == "weather-primary", "capability lookup must return weather source");
    require(!registry.find(swarx::SourceCapability::Time),
            "missing capability must return null");

    swarx::SourceRequest request;
    request.capability = swarx::SourceCapability::Weather;
    const auto response = by_capability->fetch(request);
    require(response.success, "source fetch must succeed");
    require(response.source_id == "weather-primary", "response source id must match source");
    require(response.capability == swarx::SourceCapability::Weather,
            "response capability must match source");

    require(!registry.unregister_source("does-not-exist"),
            "removing an unknown source must return false");
    require(registry.unregister_source("weather-primary"),
            "registered source must be removable");
    require(!registry.find("weather-primary"), "removed source must no longer be discoverable");
    require(registry.sources().empty(), "registry must be empty after removal");

    return EXIT_SUCCESS;
}
