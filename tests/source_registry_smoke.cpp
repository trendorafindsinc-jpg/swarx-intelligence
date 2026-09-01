#include "swarx/source_registry.hpp"

#include <cassert>
#include <memory>

namespace {
class FakeSource final : public swarx::Source {
public:
    FakeSource(std::string name, swarx::SourceCapability cap) : name_(std::move(name)), cap_(cap) {}
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
}

int main() {
    swarx::SourceRegistry registry;
    assert(registry.register_source(std::make_shared<FakeSource>("weather-primary", swarx::SourceCapability::Weather)));
    assert(!registry.register_source(std::make_shared<FakeSource>("weather-primary", swarx::SourceCapability::Weather)));
    assert(registry.find("weather-primary"));
    assert(registry.find(swarx::SourceCapability::Weather));
    assert(!registry.find(swarx::SourceCapability::Time));

    swarx::SourceRequest request;
    request.capability = swarx::SourceCapability::Weather;
    auto source = registry.find(request.capability);
    auto response = source->fetch(request);
    assert(response.success);
    assert(response.source_id == "weather-primary");
    assert(registry.unregister_source("weather-primary"));
    assert(!registry.find("weather-primary"));
    return 0;
}
