#pragma once

#include "swarx/source_trust.hpp"
#include <cstddef>
#include <string>

namespace swarx {

enum class SourceCapability { TrendoraFinds, Time, Weather, GeneralWeb };

struct SourceRequest {
    SourceCapability capability = SourceCapability::GeneralWeb;
    std::string query;
    std::string location;
    double latitude = 0.0;
    double longitude = 0.0;
    std::string timezone = "UTC";
    std::size_t max_bytes = 2 * 1024 * 1024;
    SourcePolicy policy{};
};

} // namespace swarx
