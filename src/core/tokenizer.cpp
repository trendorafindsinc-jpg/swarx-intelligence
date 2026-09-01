#include "swarx/tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace swarx {
namespace {

// SentencePiece training uses NFKC + dummy prefix; for BPE vocab pieces the
// leading "\xE2\x96\x81" (U+2581) marks a word boundary / space.
constexpr char kSpaceMarker[] = "\xE2\x96\x81"; // ▁

std::string normalize_for_bpe(std::string_view text) {
    std::string out = kSpaceMarker;
    for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            out += kSpaceMarker;
        } else {
            out.push_back(c);
        }
    }
    return out;
}

} // namespace

Tokenizer::Tokenizer() {
    // Bootstrap specials (IDs match the SWARX model contract).
    vocabulary_.emplace("<pad>", pad());
    vocabulary_.emplace("<bos>", bos());
    vocabulary_.emplace("<eos>", eos());
    vocabulary_.emplace("<unk>", unk());
    vocabulary_.emplace("<s>", bos());   // SentencePiece BOS piece
    vocabulary_.emplace("</s>", eos());  // SentencePiece EOS piece
    reverse_vocabulary_.emplace(pad(), "<pad>");
    reverse_vocabulary_.emplace(bos(), "<bos>");
    reverse_vocabulary_.emplace(eos(), "<eos>");
    reverse_vocabulary_.emplace(unk(), "<unk>");
}

void Tokenizer::load_vocab(const std::string& vocab_path) {
    std::ifstream in(vocab_path);
    if (!in) throw std::runtime_error("cannot open tokenizer vocab: " + vocab_path);

    pieces_.clear();
    piece_to_id_.clear();

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto tab = line.find('\t');
        const std::string piece = tab == std::string::npos ? line : line.substr(0, tab);
        const auto id = static_cast<TokenId>(pieces_.size());
        pieces_.push_back(piece);
        piece_to_id_.emplace(piece, id);
    }
    if (pieces_.size() < 4)
        throw std::runtime_error("tokenizer vocab too small: " + vocab_path);
}

std::size_t Tokenizer::vocab_size() const noexcept {
    if (!pieces_.empty()) return pieces_.size();
    return vocabulary_.size();
}

std::vector<TokenId> Tokenizer::encode_whitespace(std::string_view text) const {
    std::istringstream input{std::string{text}};
    std::string piece;
    std::vector<TokenId> result;
    while (input >> piece) {
        const auto it = vocabulary_.find(piece);
        result.push_back(it == vocabulary_.end() ? unk() : it->second);
    }
    return result;
}

std::vector<TokenId> Tokenizer::encode_bpe(std::string_view text) const {
    // Greedy longest-match over the SentencePiece vocab. This is not a full
    // SentencePiece runtime, but it is deterministic and shares piece IDs with
    // the exported .vocab so SWARX-0.1-nano can be exercised end-to-end.
    const std::string normalized = normalize_for_bpe(text);
    std::vector<TokenId> result;
    std::size_t i = 0;
    while (i < normalized.size()) {
        std::size_t best_len = 0;
        TokenId best_id = unk();
        // Cap search length to keep encode O(n * max_piece).
        const std::size_t max_piece = 16;
        const std::size_t limit = std::min(normalized.size() - i, max_piece);
        for (std::size_t len = limit; len >= 1; --len) {
            const std::string candidate = normalized.substr(i, len);
            const auto it = piece_to_id_.find(candidate);
            if (it != piece_to_id_.end()) {
                best_len = len;
                best_id = it->second;
                break;
            }
        }
        if (best_len == 0) {
            // Consume one byte as unk to avoid infinite loops on unknown chars.
            result.push_back(unk());
            ++i;
        } else {
            result.push_back(best_id);
            i += best_len;
        }
    }
    return result;
}

std::vector<TokenId> Tokenizer::encode(std::string_view text, bool add_bos,
                                       bool add_eos) const {
    std::vector<TokenId> result;
    if (add_bos) result.push_back(bos());
    auto body = pieces_.empty() ? encode_whitespace(text) : encode_bpe(text);
    result.insert(result.end(), body.begin(), body.end());
    if (add_eos) result.push_back(eos());
    return result;
}

std::string Tokenizer::decode(const std::vector<TokenId>& tokens,
                              bool skip_special) const {
    std::string result;
    for (const auto token : tokens) {
        if (skip_special &&
            (token == pad() || token == bos() || token == eos() || token == unk())) {
            continue;
        }
        std::string piece;
        if (!pieces_.empty() && static_cast<std::size_t>(token) < pieces_.size()) {
            piece = pieces_[token];
        } else {
            const auto it = reverse_vocabulary_.find(token);
            if (it == reverse_vocabulary_.end()) continue;
            piece = it->second;
        }
        // Convert SentencePiece space marker back to a real space.
        if (piece.rfind(kSpaceMarker, 0) == 0) {
            if (!result.empty()) result.push_back(' ');
            result += piece.substr(sizeof(kSpaceMarker) - 1);
        } else {
            result += piece;
        }
    }
    return result;
}

} // namespace swarx
