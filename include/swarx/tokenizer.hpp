#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace swarx {

using TokenId = std::uint32_t;

// Special-token contract shared with swarx-ai-models exports:
// pad=0, bos=1, eos=2, unk=3
class Tokenizer {
public:
    Tokenizer();

    // Load a SentencePiece-style .vocab file (one "piece\tscore" per line).
    // Piece order defines token IDs. Enables SWARX-0.1-nano and later models.
    void load_vocab(const std::string& vocab_path);

    bool has_vocab() const noexcept { return !pieces_.empty(); }
    std::size_t vocab_size() const noexcept;

    std::vector<TokenId> encode(std::string_view text, bool add_bos = false,
                                bool add_eos = false) const;
    std::string decode(const std::vector<TokenId>& tokens,
                       bool skip_special = true) const;

    TokenId pad() const noexcept { return 0; }
    TokenId bos() const noexcept { return 1; }
    TokenId eos() const noexcept { return 2; }
    TokenId unk() const noexcept { return 3; }

private:
    // Bootstrap whitespace map (used only when no vocab is loaded).
    std::unordered_map<std::string, TokenId> vocabulary_;
    std::unordered_map<TokenId, std::string> reverse_vocabulary_;

    // SentencePiece-compatible pieces: index == token id.
    std::vector<std::string> pieces_;
    std::unordered_map<std::string, TokenId> piece_to_id_;

    std::vector<TokenId> encode_whitespace(std::string_view text) const;
    std::vector<TokenId> encode_bpe(std::string_view text) const;
};

} // namespace swarx
