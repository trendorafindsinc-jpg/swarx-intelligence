#pragma once

#include <string>
#include <vector>

namespace swarx {

enum class ToolCapability {
    Calculator,
    TrendoraKnowledge,
    Time,
    Weather,
    WebSearch,
    FinanceData,
    News,
    Documents,
    Vision,
    Code,
};

struct ToolDescriptor {
    std::string id;
    std::string name;
    ToolCapability capability = ToolCapability::Calculator;
    bool enabled = true;
    bool requires_network = false;
    bool requires_user_permission = false;
};

// Central capability registry. The model never receives raw network access;
// product code asks the registry which controlled capability is available.
class ToolRegistry {
public:
    ToolRegistry();

    const ToolDescriptor* find(const std::string& id) const noexcept;
    bool available(ToolCapability capability) const noexcept;
    const std::vector<ToolDescriptor>& all() const noexcept { return tools_; }

private:
    std::vector<ToolDescriptor> tools_;
};

} // namespace swarx
