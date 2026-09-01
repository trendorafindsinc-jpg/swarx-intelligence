#include "swarx/weights.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

void write_string(std::ofstream& out, const std::string& value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("SWARX tensor name is too long");
    write_value(out, static_cast<std::uint32_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!out) throw std::runtime_error("failed writing SWARX tensor name");
}

std::string read_string(std::ifstream& in) {
    const auto length = read_value<std::uint32_t>(in);
    std::string value(length, '\0');
    if (length != 0)
        in.read(value.data(), static_cast<std::streamsize>(length));
    if (!in) throw std::runtime_error("invalid or truncated SWARX tensor name");
    return value;
}

void write_tensor(std::ofstream& out, const std::string& name, const Tensor& tensor) {
    write_string(out, name);
    if (tensor.shape().size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("SWARX tensor rank is too large");
    write_value(out, static_cast<std::uint32_t>(tensor.shape().size()));
    for (const auto dim : tensor.shape())
        write_value(out, static_cast<std::uint64_t>(dim));
    write_value(out, static_cast<std::uint64_t>(tensor.size()));
    if (tensor.size() != 0) {
        out.write(reinterpret_cast<const char*>(tensor.data()),
                  static_cast<std::streamsize>(tensor.size() * sizeof(float)));
        if (!out) throw std::runtime_error("failed writing SWARX tensor data");
    }
}

struct TensorRecord {
    std::string name;
    Tensor tensor;
};

TensorRecord read_tensor(std::ifstream& in) {
    TensorRecord record;
    record.name = read_string(in);
    const auto rank = read_value<std::uint32_t>(in);
    if (rank > 64) throw std::runtime_error("invalid SWARX tensor rank");

    std::vector<std::size_t> shape;
    shape.reserve(rank);
    std::uint64_t expected_size = 1;
    for (std::uint32_t i = 0; i < rank; ++i) {
        const auto dim64 = read_value<std::uint64_t>(in);
        if (dim64 > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("SWARX tensor dimension is too large");
        const auto dim = static_cast<std::size_t>(dim64);
        if (dim != 0 && expected_size > std::numeric_limits<std::uint64_t>::max() / dim)
            throw std::runtime_error("SWARX tensor size overflow");
        expected_size *= dim;
        shape.push_back(dim);
    }

    const auto element_count = read_value<std::uint64_t>(in);
    if (element_count != expected_size)
        throw std::runtime_error("SWARX tensor element count mismatch");
    if (element_count > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("SWARX tensor is too large");

    record.tensor = Tensor(std::move(shape));
    if (record.tensor.size() != static_cast<std::size_t>(element_count))
        throw std::runtime_error("SWARX tensor allocation mismatch");
    if (record.tensor.size() != 0) {
        in.read(reinterpret_cast<char*>(record.tensor.data()),
                static_cast<std::streamsize>(record.tensor.size() * sizeof(float)));
        if (!in) throw std::runtime_error("invalid or truncated SWARX tensor data");
    }
    return record;
}

std::uint64_t expected_tensor_count(const ModelConfig& config) {
    return 4ull + 8ull * static_cast<std::uint64_t>(config.num_layers);
}

void assign_tensor(Tensor& destination, TensorRecord&& record, const std::string& expected) {
    if (record.name != expected)
        throw std::runtime_error("unexpected SWARX tensor: " + record.name + ", expected " + expected);
    if (record.tensor.shape() != destination.shape())
        throw std::runtime_error("SWARX tensor shape mismatch for " + expected);
    destination = std::move(record.tensor);
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

    write_value(out, expected_tensor_count(c));
    write_tensor(out, "token_embedding", model.token_embedding());
    write_tensor(out, "position_embedding", model.position_embedding());
    write_tensor(out, "final_norm_weight", model.final_norm_weight());
    write_tensor(out, "lm_head", model.lm_head());

    const auto& blocks = model.blocks();
    if (blocks.size() != c.num_layers)
        throw std::runtime_error("SWARX model layer count mismatch");
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        const auto prefix = "layers." + std::to_string(i) + ".";
        const auto& block = blocks[i];
        write_tensor(out, prefix + "attention.query_weight", block.attention().query_weight());
        write_tensor(out, prefix + "attention.key_weight", block.attention().key_weight());
        write_tensor(out, prefix + "attention.value_weight", block.attention().value_weight());
        write_tensor(out, prefix + "attention.output_weight", block.attention().output_weight());
        write_tensor(out, prefix + "attention_norm_weight", block.attention_norm_weight());
        write_tensor(out, prefix + "mlp_norm_weight", block.mlp_norm_weight());
        write_tensor(out, prefix + "up_weight", block.up_weight());
        write_tensor(out, prefix + "down_weight", block.down_weight());
    }
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

TransformerModel WeightsReader::load(const std::string& path) {
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

    const auto count = read_value<std::uint64_t>(in);
    if (count != expected_tensor_count(c))
        throw std::runtime_error("SWARX tensor count does not match model configuration");

    TransformerModel model(c);
    assign_tensor(model.token_embedding_, read_tensor(in), "token_embedding");
    assign_tensor(model.position_embedding_, read_tensor(in), "position_embedding");
    assign_tensor(model.final_norm_weight_, read_tensor(in), "final_norm_weight");
    assign_tensor(model.lm_head_, read_tensor(in), "lm_head");

    for (std::size_t i = 0; i < c.num_layers; ++i) {
        const auto prefix = "layers." + std::to_string(i) + ".";
        auto& block = model.blocks_[i];
        assign_tensor(block.attention_.query_weight_, read_tensor(in), prefix + "attention.query_weight");
        assign_tensor(block.attention_.key_weight_, read_tensor(in), prefix + "attention.key_weight");
        assign_tensor(block.attention_.value_weight_, read_tensor(in), prefix + "attention.value_weight");
        assign_tensor(block.attention_.output_weight_, read_tensor(in), prefix + "attention.output_weight");
        assign_tensor(block.attention_norm_weight_, read_tensor(in), prefix + "attention_norm_weight");
        assign_tensor(block.mlp_norm_weight_, read_tensor(in), prefix + "mlp_norm_weight");
        assign_tensor(block.up_weight_, read_tensor(in), prefix + "up_weight");
        assign_tensor(block.down_weight_, read_tensor(in), prefix + "down_weight");
    }

    return model;
}

} // namespace swarx
