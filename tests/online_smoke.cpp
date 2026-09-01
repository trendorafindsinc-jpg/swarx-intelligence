#include "swarx/online.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>

int main() {
    using namespace swarx;

    OnlineInformationGateway gateway;

    OnlineRequest weather;
    weather.intent = OnlineIntent::Weather;
    weather.latitude = 100.0;
    weather.longitude = 3.3792;
    weather.timezone = "Africa/Lagos";
    const auto invalid = gateway.fetch(weather);
    assert(!invalid.success);
    assert(invalid.error.find("invalid latitude") != std::string::npos);

    // Network access is deliberately opt-in. Normal test runs never spend
    // the developer's mobile data or depend on an external service.
    if (std::getenv("SWARX_RUN_ONLINE_TESTS") != nullptr) {
        OnlineRequest time;
        time.intent = OnlineIntent::Time;
        time.timezone = "Africa/Lagos";
        const auto live = gateway.fetch(time);
        assert(live.success);
        assert(!live.source.body.empty());
        assert(live.source.status_code == 200);
    }

    OnlineRequest custom;
    custom.intent = OnlineIntent::Custom;
    custom.custom_url = "ftp://invalid.example";
    const auto invalid_scheme = gateway.fetch(custom);
    assert(!invalid_scheme.success);
    assert(invalid_scheme.error.find("only permits") != std::string::npos);

    std::cout << "SWARX online gateway smoke test passed\n";
    return 0;
}
