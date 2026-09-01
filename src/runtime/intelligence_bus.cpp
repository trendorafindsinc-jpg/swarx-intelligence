#include "swarx/intelligence_bus.hpp"

#include <algorithm>
#include <utility>

namespace swarx {

std::uint64_t IntelligenceBus::begin_request(const std::string& task) {
    const auto id = next_request_id_++;
    IntelligenceEnvelope envelope;
    envelope.request_id = id;
    envelope.kind = BusMessageKind::Request;
    envelope.sender = "swarx-router";
    envelope.task = task;
    messages_.push_back(std::move(envelope));
    return id;
}

bool IntelligenceBus::publish(IntelligenceEnvelope message) {
    if (message.request_id == 0 || message.sender.empty() || message.task.empty()) return false;
    if (message.kind == BusMessageKind::Request) return false;
    if (message.confidence < 0.0 || message.confidence > 1.0) return false;

    const auto request_exists = std::any_of(
        messages_.begin(), messages_.end(),
        [&message](const IntelligenceEnvelope& item) { return item.request_id == message.request_id; });
    if (!request_exists) return false;

    messages_.push_back(std::move(message));
    return true;
}

std::vector<IntelligenceEnvelope> IntelligenceBus::messages(std::uint64_t request_id) const {
    std::vector<IntelligenceEnvelope> result;
    for (const auto& message : messages_) {
        if (message.request_id == request_id) result.push_back(message);
    }
    return result;
}

BusTrace IntelligenceBus::trace(std::uint64_t request_id) const {
    BusTrace result;
    result.request_id = request_id;
    for (const auto& message : messages_) {
        if (message.request_id != request_id) continue;
        ++result.message_count;
        if (!message.sender.empty() &&
            std::find(result.participants.begin(), result.participants.end(), message.sender) == result.participants.end()) {
            result.participants.push_back(message.sender);
        }
        for (const auto& source : message.source_ids) {
            if (std::find(result.sources.begin(), result.sources.end(), source) == result.sources.end()) {
                result.sources.push_back(source);
            }
        }
        if (message.kind == BusMessageKind::Error) result.has_error = true;
    }
    return result;
}

void IntelligenceBus::clear() {
    messages_.clear();
    next_request_id_ = 1;
}

} // namespace swarx
