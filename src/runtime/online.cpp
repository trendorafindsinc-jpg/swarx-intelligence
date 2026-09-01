#include "swarx/online.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace swarx {
namespace {

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&time, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string url_encode(std::string_view value) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    for (const unsigned char c : value) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 0x0F]);
        }
    }
    return out;
}

std::string append_query(const std::string& base, const std::string& key, const std::string& value) {
    return base + (base.find('?') == std::string::npos ? '?' : '&') + key + '=' + url_encode(value);
}

} // namespace

OnlineInformationGateway::OnlineInformationGateway(OnlineConfig config)
    : config_(std::move(config)) {}

OnlineResult OnlineInformationGateway::fetch_url(const std::string& source_id,
                                                   const std::string& url) const {
    OnlineResult result;
    result.source_id = source_id;
    result.source.source_id = source_id;
    result.source.url = url;
    result.source.retrieved_at_utc = utc_now();

    HttpRequest request;
    request.url = url;
    request.timeout_seconds = config_.timeout_seconds;
    request.max_bytes = config_.max_response_bytes;
    request.headers.emplace_back("Accept: application/json, text/plain;q=0.9, */*;q=0.5");
    const auto response = client_.get(request);
    result.source.status_code = response.status_code;
    result.source.body = response.body;
    if (!response.ok()) {
        result.error = response.error.empty() ? "online source request failed" : response.error;
        return result;
    }
    result.success = true;
    result.answer_hint = "Fresh information retrieved from " + source_id + ".";
    return result;
}

OnlineResult OnlineInformationGateway::fetch(const OnlineRequest& request) const {
    switch (request.intent) {
        case OnlineIntent::Trendora: {
            auto url = config_.trendora_endpoint;
            if (!request.query.empty()) {
                url = append_query(url, "search", request.query);
            }
            return fetch_url("trendora", url);
        }
        case OnlineIntent::Time: {
            if (request.timezone.empty()) {
                return {false, "timeapi", {}, "timezone cannot be empty", {}};
            }
            auto url = append_query(config_.time_endpoint, "timeZone", request.timezone);
            return fetch_url("timeapi", url);
        }
        case OnlineIntent::Weather: {
            if (request.latitude < -90.0 || request.latitude > 90.0 ||
                request.longitude < -180.0 || request.longitude > 180.0) {
                return {false, "open-meteo", {}, "invalid latitude/longitude", {}};
            }
            auto url = append_query(config_.weather_endpoint, "latitude", std::to_string(request.latitude));
            url = append_query(url, "longitude", std::to_string(request.longitude));
            url = append_query(url, "current", "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m");
            url = append_query(url, "daily", "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,sunrise,sunset");
            url = append_query(url, "forecast_days", "7");
            if (!request.timezone.empty()) url = append_query(url, "timezone", request.timezone);
            return fetch_url("open-meteo", url);
        }
        case OnlineIntent::Custom:
            if (request.custom_url.empty()) {
                return {false, "custom", {}, "custom URL cannot be empty", {}};
            }
            return fetch_url("custom", request.custom_url);
    }
    throw std::logic_error("unknown SWARX online intent");
}

} // namespace swarx
