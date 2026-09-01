#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace swarx {

using TokenId = std::uint32_t;

class Tokenizer {
public:
    Tokenizer();

    std::vector<TokenId> encode(std::string_view text) const;
    std::string decode(const std::vector<TokenId>& tokens) const;

    TokenId pad() const noexcept { return 0; }
    TokenId bos() const noexcept { return 1; }
    TokenId eos() const noexcept { return 2; }
    TokenId unk() const noexcept { return 3; }

private:
    std::unordered_map<std::string, TokenId> vocabulary_;
    std::unordered_map<TokenId, std::string> reverse_vocabulary_;
};

} // namespace swarx
