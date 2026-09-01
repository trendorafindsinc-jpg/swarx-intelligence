#pragma once

#include "swarx/source_request.hpp"
#include <string>

namespace swarx {

struct SourceResponse {
    bool success = false;
    std::string source_id;
    SourceCapability capability = SourceCapability::GeneralWeb;
    std::string url;
    long status_code = 0;
    std::string content_type;
    std::string body;
    SourceEvidence evidence{};
    std::string error;
};

} // namespace swarx
