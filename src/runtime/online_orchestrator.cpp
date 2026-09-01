#include "swarx/online_orchestrator.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <utility>

namespace swarx {
namespace {

bool contains_ci(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

} // namespace

OnlineIntelligenceOrchestrator::OnlineIntelligenceOrchestrator(OnlineOrchestratorConfig config)
    : config_(std::move(config)), knowledge_(config_.knowledge) {}

std::string OnlineIntelligenceOrchestrator::normalize(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool previous_space = false;
    for (unsigned char c : text) {
        if (std::isspace(c)) {
            if (!previous_space) out.push_back(' ');
            previous_space = true;
        } else {
            out.push_back(static_cast<char>(std::tolower(c)));
            previous_space = false;
        }
    }
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

bool OnlineIntelligenceOrchestrator::contains_any(const std::string& text,
                                                    const char* const* terms,
                                                    std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (contains_ci(text, terms[i])) return true;
    }
    return false;
}

OnlinePlan OnlineIntelligenceOrchestrator::plan(const OnlineOrchestrationRequest& request) const {
    const std::string q = normalize(request.query);
    OnlinePlan result;

    static constexpr const char* time_terms[] = {
        "what time", "current time", "time in ", "time now", "local time"
    };
    static constexpr const char* weather_terms[] = {
        "weather", "forecast", "temperature", "rain today", "rain tomorrow"
    };
    static constexpr const char* fresh_terms[] = {
        "today", "tonight", "now", "latest", "current", "recent", "news", "this week"
    };
    static constexpr const char* finance_terms[] = {
        "finance", "money", "investment", "investing", "stock", "budget", "saving",
        "savings", "interest", "loan", "debt", "financial"
    };
    static constexpr const char* education_terms[] = {
        "learn", "explain", "education", "lesson", "study", "teach", "meaning", "how does"
    };

    if (q.empty()) {
        result.need = OnlineNeed::None;
        result.reason = "empty request does not require online information";
        return result;
    }

    if (contains_any(q, time_terms, sizeof(time_terms) / sizeof(time_terms[0]))) {
        result.need = OnlineNeed::Time;
        result.route = KnowledgeRoute::Time;
        result.requires_freshness = true;
        result.reason = "time is live information and must come from a time service";
        return result;
    }

    if (contains_any(q, weather_terms, sizeof(weather_terms) / sizeof(weather_terms[0]))) {
        result.need = OnlineNeed::Weather;
        result.route = KnowledgeRoute::Weather;
        result.requires_freshness = true;
        result.reason = "weather changes over time and requires a live forecast source";
        return result;
    }

    if (contains_any(q, finance_terms, sizeof(finance_terms) / sizeof(finance_terms[0])) ||
        contains_any(q, education_terms, sizeof(education_terms) / sizeof(education_terms[0]))) {
        result.need = OnlineNeed::Knowledge;
        result.route = KnowledgeRoute::TrendoraFinds;
        result.domain = contains_any(q, finance_terms, sizeof(finance_terms) / sizeof(finance_terms[0]))
                            ? KnowledgeDomain::Finance
                            : KnowledgeDomain::Education;
        result.requires_freshness = contains_any(q, fresh_terms, sizeof(fresh_terms) / sizeof(fresh_terms[0]));
        result.reason = "finance/education request is eligible for the TrendoraFinds knowledge hub";
        return result;
    }

    if (contains_any(q, fresh_terms, sizeof(fresh_terms) / sizeof(fresh_terms[0]))) {
        result.need = OnlineNeed::GeneralWeb;
        result.route = KnowledgeRoute::GeneralWeb;
        result.requires_freshness = true;
        result.reason = "request explicitly asks for current information";
        return result;
    }

    result.need = OnlineNeed::Knowledge;
    result.route = KnowledgeRoute::GeneralWeb;
    result.reason = "general factual request can use the general knowledge route";
    return result;
}

std::string OnlineIntelligenceOrchestrator::cache_key(
    const OnlineOrchestrationRequest& request, const OnlinePlan& plan) {
    std::ostringstream key;
    key << normalize(request.query) << '|' << request.location << '|'
        << request.latitude << '|' << request.longitude << '|' << request.timezone << '|'
        << static_cast<int>(plan.route) << '|' << plan.requires_freshness;
    return key.str();
}

void OnlineIntelligenceOrchestrator::trim_cache_locked() const {
    while (cache_.size() > config_.max_cache_entries) {
        auto oldest = cache_.begin();
        for (auto it = cache_.begin(); it != cache_.end(); ++it) {
            if (it->second.expires_at < oldest->second.expires_at) oldest = it;
        }
        cache_.erase(oldest);
    }
}

OnlineOrchestrationResult OnlineIntelligenceOrchestrator::execute(
    const OnlineOrchestrationRequest& request) const {
    OnlineOrchestrationResult output;
    output.plan = plan(request);

    if (output.plan.need == OnlineNeed::None) {
        output.success = false;
        output.error = "online orchestration requires a non-empty request";
        return output;
    }
    if (!request.allow_network && !request.allow_cache) {
        output.error = "both network and cache access are disabled";
        return output;
    }

    const std::string key = cache_key(request, output.plan);
    const auto now = std::chrono::steady_clock::now();

    if (request.allow_cache) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        const auto it = cache_.find(key);
        if (it != cache_.end()) {
            if (it->second.expires_at > now) {
                output.success = it->second.result.success;
                output.from_cache = true;
                output.knowledge = it->second.result;
                output.error = it->second.result.error;
                return output;
            }
            cache_.erase(it);
        }
    }

    if (!request.allow_network) {
        output.error = "network access disabled and no fresh cache entry is available";
        return output;
    }

    KnowledgeQuery query;
    query.text = request.query;
    query.domain = output.plan.domain;
    query.require_fresh = output.plan.requires_freshness;
    query.max_sources = request.max_sources;
    query.route = output.plan.route;
    query.location = request.location;
    query.latitude = request.latitude;
    query.longitude = request.longitude;
    query.timezone = request.timezone;

    output.knowledge = knowledge_.retrieve(query);
    output.success = output.knowledge.success;
    output.error = output.knowledge.error;

    if (output.success && request.allow_cache && config_.max_cache_entries > 0 &&
        config_.cache_ttl.count() > 0) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_[key] = CacheEntry{output.knowledge, now + config_.cache_ttl};
        trim_cache_locked();
    }

    return output;
}

void OnlineIntelligenceOrchestrator::clear_cache() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
}

std::size_t OnlineIntelligenceOrchestrator::cache_size() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_.size();
}

} // namespace swarx
