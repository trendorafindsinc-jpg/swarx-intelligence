#include "swarx/weights.hpp"

#include <fstream>
#include <stdexcept>

namespace swarx {
namespace {

template <typename T>
void write_value(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed writing SWARX weights file");
}

template <typename T>
T read_value(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("invalid or truncated SWARX weights file");
    return value;
}

} // namespace

void WeightsWriter::write(const std::string& path, const TransformerModel& model) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create SWARX weights file: " + path);

    const auto& c = model.config();
    write_value(out, kWeightsMagic);
    write_value(out, kWeightsVersion);
    write_value(out, static_cast<std::uint64_t>(c.vocab_size));
    write_value(out, static_cast<std::uint64_t>(c.hidden_size));
    write_value(out, static_cast<std::uint64_t>(c.intermediate_size));
    write_value(out, static_cast<std::uint64_t>(c.num_heads));
    write_value(out, static_cast<std::uint64_t>(c.num_layers));
    write_value(out, static_cast<std::uint64_t>(c.max_sequence_length));
    write_value(out, c.rms_epsilon);

    // v1 is deliberately a metadata-only container. Weight serialization will
    // be added after tensor ownership/loading is exposed by the model API.
    write_value(out, static_cast<std::uint64_t>(0));
}

WeightsFileInfo WeightsReader::inspect(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open SWARX weights file: " + path);

    if (read_value<std::uint32_t>(in) != kWeightsMagic)
        throw std::runtime_error("invalid SWARX weights magic");
    if (read_value<std::uint32_t>(in) != kWeightsVersion)
        throw std::runtime_error("unsupported SWARX weights version");

    ModelConfig c;
    c.vocab_size = static_cast<std::size_t>(read_value<std::uint64_t>(in));
    c.hidden_size = static_cast<std::size_t>(read_value<std::uint64_t>(in));
    c.intermediate_size = static_cast<std::size_t>(read_value<std::uint64_t>(in));
    c.num_heads = static_cast<std::size_t>(read_value<std::uint64_t>(in));
    c.num_layers = static_cast<std::size_t>(read_value<std::uint64_t>(in));
    c.max_sequence_length = static_cast<std::size_t>(read_value<std::uint64_t>(in));
    c.rms_epsilon = read_value<float>(in);

    WeightsFileInfo info;
    info.config = c;
    info.tensor_count = read_value<std::uint64_t>(in);
    return info;
}

} // namespace swarx
