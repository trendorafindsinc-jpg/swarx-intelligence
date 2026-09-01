#include "swarx/tokenizer.hpp"

#include <sstream>

namespace swarx {

Tokenizer::Tokenizer() {
    vocabulary_.emplace("<unk>", unk());
    vocabulary_.emplace("<bos>", bos());
    vocabulary_.emplace("<eos>", eos());
    reverse_vocabulary_.emplace(unk(), "<unk>");
    reverse_vocabulary_.emplace(bos(), "<bos>");
    reverse_vocabulary_.emplace(eos(), "<eos>");
}

std::vector<TokenId> Tokenizer::encode(std::string_view text) const {
    // Bootstrap tokenizer: whitespace units are mapped when present in the
    // vocabulary. Unknown pieces remain deterministic and map to <unk>.
    std::istringstream input{std::string{text}};
    std::string piece;
    std::vector<TokenId> result;
    while (input >> piece) {
        const auto it = vocabulary_.find(piece);
        result.push_back(it == vocabulary_.end() ? unk() : it->second);
    }
    return result;
}

std::string Tokenizer::decode(const std::vector<TokenId>& tokens) const {
    std::string result;
    for (const auto token : tokens) {
        const auto it = reverse_vocabulary_.find(token);
        if (it == reverse_vocabulary_.end()) continue;
        if (!result.empty()) result.push_back(' ');
        result += it->second;
    }
    return result;
}

} // namespace swarx
