#include "swarx/source_adapters.hpp"

#include <cassert>
#include <string>

int main() {
    using namespace swarx;

    const auto registry = make_default_source_registry();
    const auto& sources = registry.sources();
    assert(!sources.empty());

    bool found_trendora = false;
    bool found_time = false;
    bool found_weather = false;
    bool found_general = false;

    for (const auto& source : sources) {
        assert(source != nullptr);
        const auto& id = source->id();
        if (id == "trendorafinds") found_trendora = true;
        if (id == "timeapi") found_time = true;
        if (id == "open-meteo") found_weather = true;
        if (id == "wikipedia") found_general = true;
    }

    assert(found_trendora);
    assert(found_time);
    assert(found_weather);
    assert(found_general);
    return 0;
}
