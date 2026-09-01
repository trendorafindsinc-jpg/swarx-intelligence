#include "swarx/online.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>

int main() {
    using namespace swarx;

    OnlineInformationGateway gateway;

    OnlineRequest weather;
    weather.intent = OnlineIntent::Weather;
    weather.latitude = 6.5244;
    weather.longitude = 3.3792;
    weather.timezone = "Africa/Lagos";

    // Structural validation must work without network access.
    weather.latitude = 100.0;
    const auto invalid = gateway.fetch(weather);
    assert(!invalid.success);
    assert(invalid.error.find("invalid latitude") != std::string::npos);

    OnlineRequest time;
    time.intent = OnlineIntent::Time;
    time.timezone = "Africa/Lagos";
    const auto valid_time_request = gateway.fetch(time);
    // Network smoke is opt-in so CI/offline builds remain deterministic.
    if (std::getenv("SWARX_RUN_ONLINE_TESTS") != nullptr) {
        assert(valid_time_request.success);
        assert(!valid_time_request.source.body.empty());
        assert(valid_time_request.source.status_code == 200);
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
