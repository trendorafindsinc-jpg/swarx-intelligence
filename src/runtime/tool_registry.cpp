#include "swarx/tool_registry.hpp"

namespace swarx {

ToolRegistry::ToolRegistry()
    : tools_{
          {"calculator", "Calculator", ToolCapability::Calculator, true, false, false},
          {"trendora-knowledge", "TrendoraFinds Knowledge", ToolCapability::TrendoraKnowledge, true, true, false},
          {"time", "Live Time", ToolCapability::Time, true, true, false},
          {"weather", "Weather", ToolCapability::Weather, true, true, false},
          {"web-search", "Web Search", ToolCapability::WebSearch, false, true, true},
          {"finance-data", "Finance Data", ToolCapability::FinanceData, false, true, true},
          {"news", "News", ToolCapability::News, false, true, true},
          {"documents", "Documents", ToolCapability::Documents, false, false, true},
          {"vision", "Vision", ToolCapability::Vision, false, false, true},
          {"code", "Code Execution", ToolCapability::Code, false, false, true},
      } {}

const ToolDescriptor* ToolRegistry::find(const std::string& id) const noexcept {
    for (const auto& tool : tools_) {
        if (tool.id == id) return &tool;
    }
    return nullptr;
}

bool ToolRegistry::available(ToolCapability capability) const noexcept {
    for (const auto& tool : tools_) {
        if (tool.capability == capability) return tool.enabled;
    }
    return false;
}

} // namespace swarx
