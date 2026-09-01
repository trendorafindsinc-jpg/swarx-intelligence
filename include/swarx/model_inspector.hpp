#pragma once

#include "swarx/model.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace swarx {

struct ModelReport {
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t intermediate_size = 0;
    std::size_t num_heads = 0;
    std::size_t num_layers = 0;
    std::size_t max_sequence_length = 0;
    std::size_t parameter_count = 0;
    std::size_t parameter_bytes = 0;
    std::uint64_t fingerprint = 0;
    bool finite = true;
    std::vector<std::string> warnings;
};

class ModelInspector {
public:
    static ModelReport inspect(const TransformerModel& model);
    static void validate(const TransformerModel& model);
};

} // namespace swarx
