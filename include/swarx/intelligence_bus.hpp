#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace swarx {

enum class BusMessageKind {
    Request,
    Observation,
    ModelResponse,
    ToolResult,
    Verification,
    FinalResponse,
    Error,
};

struct CapabilityRequirement {
    std::string name;
    bool required = true;
};

struct IntelligenceEnvelope {
    std::uint64_t request_id = 0;
    std::uint64_t parent_id = 0;
    BusMessageKind kind = BusMessageKind::Request;
    std::string sender;
    std::string task;
    std::string payload;
    std::vector<CapabilityRequirement> capabilities;
    std::vector<std::string> source_ids;
    double confidence = 0.0;
    bool verified = false;
};

struct BusTrace {
    std::uint64_t request_id = 0;
    std::size_t message_count = 0;
    std::vector<std::string> participants;
    std::vector<std::string> sources;
    bool has_error = false;
};

// Lightweight structured communication contract for model/tool cooperation.
// The bus stores envelopes and provenance; it does not execute arbitrary code.
class IntelligenceBus {
public:
    IntelligenceBus() = default;

    std::uint64_t begin_request(const std::string& task);
    bool publish(IntelligenceEnvelope message);
    std::vector<IntelligenceEnvelope> messages(std::uint64_t request_id) const;
    BusTrace trace(std::uint64_t request_id) const;
    void clear();

private:
    std::uint64_t next_request_id_ = 1;
    std::vector<IntelligenceEnvelope> messages_;
};

} // namespace swarx
