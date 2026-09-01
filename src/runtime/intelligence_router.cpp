#include "swarx/intelligence_router.hpp"

#include <algorithm>
#include <cctype>
#include <utility>

namespace swarx {
namespace {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool contains_any(const std::string& text, std::initializer_list<const char*> words) {
    for (const auto* word : words) {
        if (text.find(word) != std::string::npos) return true;
    }
    return false;
}

} // namespace

IntelligenceRouter::IntelligenceRouter(IntelligenceRouterConfig config)
    : config_(std::move(config)), online_(config_.online) {
    sources_ = {
        {"trendora", "TrendoraFinds knowledge hub", SourceClass::OwnedKnowledge, true, true, 100},
        {"timeapi", "Time service", SourceClass::LiveUtility, true, true, 90},
        {"open-meteo", "Open-Meteo weather service", SourceClass::LiveUtility, true, true, 90},
        {"web", "External web/search adapter", SourceClass::ExternalWeb, config_.allow_live_web, true, 50},
        {"finance", "External finance data adapter", SourceClass::ExternalWeb, config_.allow_finance_data, true, 60},
        {"news", "External news adapter", SourceClass::ExternalWeb, config_.allow_news, true, 50},
        {"calculator", "Local deterministic computation", SourceClass::Computation, true, false, 80},
    };
}

IntelligenceIntent IntelligenceRouter::classify(const std::string& raw) const {
    const auto text = lower(raw);

    if (contains_any(text, {"trendorafinds", "trendora finds", "trendora", "from my knowledge hub"})) {
        return IntelligenceIntent::TrendoraKnowledge;
    }
    if (contains_any(text, {"what time", "current time", "time in ", "time now", "timezone"})) {
        return IntelligenceIntent::Time;
    }
    if (contains_any(text, {"weather", "forecast", "temperature", "rain today", "rain tomorrow", "humidity"})) {
        return IntelligenceIntent::Weather;
    }
    if (contains_any(text, {"news", "latest news", "breaking", "today's events", "what happened today"})) {
        return IntelligenceIntent::News;
    }
    if (contains_any(text, {"stock price", "share price", "exchange rate", "forex", "market price", "crypto price"})) {
        return IntelligenceIntent::FinanceData;
    }
    if (contains_any(text, {"calculate", "what is ", "how much is ", "multiply", "divide", "plus", "minus"}) &&
        contains_any(text, {"+", "-", "*", "/", "times", "percent", "calculate", "how much"})) {
        return IntelligenceIntent::Calculator;
    }
    if (contains_any(text, {"search the web", "search online", "look online", "look this up", "latest information", "on the internet"})) {
        return IntelligenceIntent::LiveWeb;
    }
    return IntelligenceIntent::General;
}

RouteDecision IntelligenceRouter::route_for(IntelligenceIntent intent) const {
    switch (intent) {
        case IntelligenceIntent::TrendoraKnowledge:
            return {intent, SourceClass::OwnedKnowledge, "trendora",
                    "Use SWARX-owned TrendoraFinds knowledge for finance, education and related company content.", true, true};
        case IntelligenceIntent::Time:
            return {intent, SourceClass::LiveUtility, "timeapi",
                    "Time is volatile; use a live time source instead of model memory.", true, true};
        case IntelligenceIntent::Weather:
            return {intent, SourceClass::LiveUtility, "open-meteo",
                    "Weather is volatile; use live forecast data instead of model memory.", true, true};
        case IntelligenceIntent::FinanceData:
            return {intent, SourceClass::ExternalWeb, "finance",
                    "Financial market data must come from a live provider when enabled.", true, config_.allow_finance_data};
        case IntelligenceIntent::News:
            return {intent, SourceClass::ExternalWeb, "news",
                    "News is time-sensitive and requires a live source when enabled.", true, config_.allow_news};
        case IntelligenceIntent::LiveWeb:
            return {intent, SourceClass::ExternalWeb, "web",
                    "General current information requires an explicitly enabled web adapter.", true, config_.allow_live_web};
        case IntelligenceIntent::Calculator:
            return {intent, SourceClass::Computation, "calculator",
                    "Deterministic arithmetic should not consume model or network resources.", false, true};
        case IntelligenceIntent::General:
            return {intent, SourceClass::Unavailable, "local-model",
                    "No external source is required by the initial classifier.", false, true};
    }
    return {};
}

RouteDecision IntelligenceRouter::decide(const IntelligenceRequest& request) const {
    auto decision = route_for(classify(request.text));
    if (decision.requires_network && (!config_.network_enabled || !request.network_allowed)) {
        decision.supported = false;
        decision.rationale += " Network access is disabled by policy.";
    }
    return decision;
}

IntelligenceResult IntelligenceRouter::execute(const IntelligenceRequest& request) const {
    IntelligenceResult result;
    result.route = decide(request);

    if (!result.route.supported) {
        result.error = "selected intelligence source is unavailable under current policy";
        return result;
    }

    if (result.route.intent == IntelligenceIntent::General ||
        result.route.intent == IntelligenceIntent::Calculator) {
        result.success = true;
        result.answer_hint = result.route.intent == IntelligenceIntent::Calculator
            ? "Local computation capability selected."
            : "No online source required; continue with the local SWARX model/runtime.";
        return result;
    }

    if (!request.network_allowed || !config_.network_enabled) {
        result.error = "network access denied";
        return result;
    }

    OnlineRequest online_request;
    online_request.query = request.text;
    online_request.location = request.location;
    online_request.timezone = request.timezone;
    online_request.latitude = request.latitude;
    online_request.longitude = request.longitude;

    switch (result.route.intent) {
        case IntelligenceIntent::TrendoraKnowledge:
            online_request.intent = OnlineIntent::Trendora;
            break;
        case IntelligenceIntent::Time:
            online_request.intent = OnlineIntent::Time;
            break;
        case IntelligenceIntent::Weather:
            online_request.intent = OnlineIntent::Weather;
            break;
        default:
            result.error = "selected adapter is registered but not yet implemented";
            return result;
    }

    const auto online_result = online_.fetch(online_request);
    result.success = online_result.success;
    result.answer_hint = online_result.answer_hint;
    result.error = online_result.error;
    if (online_result.success) result.sources.push_back(online_result.source);
    return result;
}

} // namespace swarx
