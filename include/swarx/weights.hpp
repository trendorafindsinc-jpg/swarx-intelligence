#pragma once

#include "swarx/model.hpp"
#include <cstdint>
#include <string>

namespace swarx {

inline constexpr std::uint32_t kWeightsMagic = 0x53574152u; // "SWAR"
inline constexpr std::uint32_t kWeightsVersion = 1;

struct WeightsFileInfo {
    ModelConfig config{};
    std::uint64_t tensor_count = 0;
};

class WeightsWriter {
public:
    static void write(const std::string& path, const TransformerModel& model);
};

class WeightsReader {
public:
    static WeightsFileInfo inspect(const std::string& path);
};

} // namespace swarx
