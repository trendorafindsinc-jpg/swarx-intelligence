#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace swarx {

struct HttpRequest {
    std::string url;
    std::size_t timeout_seconds = 10;
    std::size_t max_bytes = 2 * 1024 * 1024;
    std::vector<std::string> headers;
};

struct HttpResponse {
    long status_code = 0;
    std::string body;
    std::string error;
    bool ok() const noexcept { return error.empty() && status_code >= 200 && status_code < 300; }
};

// Dependency-light HTTPS transport for the mobile/runtime layer. It delegates
// TLS and HTTP protocol handling to the system curl executable already common
// on Termux/Linux, while keeping the SWARX API independent of curl headers.
class HttpClient {
public:
    HttpResponse get(const HttpRequest& request) const;
};

} // namespace swarx
