#pragma once

#include "swarx/web.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace swarx {

enum class OnlineIntent {
    Trendora,
    Time,
    Weather,
    Custom,
};

struct OnlineRequest {
    OnlineIntent intent = OnlineIntent::Custom;
    std::string query;
    std::string location;
    double latitude = 0.0;
    double longitude = 0.0;
    std::string timezone = "UTC";
    std::string custom_url;
};

struct SourceRecord {
    std::string source_id;
    std::string url;
    std::string content_type = "application/json";
    long status_code = 0;
    std::string retrieved_at_utc;
    std::string body;
};

struct OnlineResult {
    bool success = false;
    std::string source_id;
    std::string answer_hint;
    std::string error;
    SourceRecord source;
};

struct OnlineConfig {
    std::string trendora_endpoint =
        "https://trendorafinds.wordpress.com/wp-json/wp/v2/posts?per_page=10";
    std::string time_endpoint = "https://timeapi.io/api/time/current/zone";
    std::string weather_endpoint = "https://api.open-meteo.com/v1/forecast";
    std::size_t timeout_seconds = 8;
    std::size_t max_response_bytes = 2 * 1024 * 1024;
};

// SWARX's controlled gateway to live information. The intelligence layer can
// ask for a capability instead of hard-coding websites throughout the model.
// Every successful result carries provenance so product layers can disclose
// where fresh information came from.
class OnlineInformationGateway {
public:
    explicit OnlineInformationGateway(OnlineConfig config = {});

    OnlineResult fetch(const OnlineRequest& request) const;

private:
    OnlineResult fetch_url(const std::string& source_id, const std::string& url) const;
    OnlineConfig config_;
    HttpClient client_;
};

} // namespace swarx
