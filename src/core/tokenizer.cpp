#include "swarx/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace swarx {
namespace {

// SentencePiece's default text representation uses U+2581 ("▁") as the
// escaped whitespace marker and adds a dummy marker at the start of non-empty
// input. The tokenizer trainer currently uses SentencePiece BPE defaults.
constexpr char kSpaceMarker[] = "\xE2\x96\x81"; // ▁

bool is_whitespace_cp(std::uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v';
}

std::string normalize_for_bpe(std::string_view text) {
    if (text.empty()) return {};

    // SentencePiece's default normalizer removes leading/trailing whitespace
    // and collapses a whitespace run to one logical space before escaping it
    // as ▁. Keeping this normalization here makes the native path agree with
    // the Python SentencePieceProcessor without requiring the C++ library.
    std::string out = kSpaceMarker;
    bool pending_space = false;
    bool emitted_non_space = false;

    for (std::size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        std::uint32_t cp = 0;
        std::size_t width = 1;
        if (c < 0x80) {
            cp = c;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            cp = ((c & 0x1F) << 6) |
                 (static_cast<unsigned char>(text[i + 1]) & 0x3F);
            width = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            cp = ((c & 0x0F) << 12) |
                 ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(text[i + 2]) & 0x3F);
            width = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) {
            cp = ((c & 0x07) << 18) |
                 ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
                 ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(text[i + 3]) & 0x3F);
            width = 4;
        } else {
            cp = 0xFFFD;
        }

        if (is_whitespace_cp(cp)) {
            if (emitted_non_space) pending_space = true;
        } else {
            if (pending_space) {
                out += kSpaceMarker;
                pending_space = false;
            }
            out.append(text.substr(i, width));
            emitted_non_space = true;
        }
        i += width;
    }

    if (!emitted_non_space) return {};
    return out;
}

bool next_utf8_codepoint(std::string_view text, std::size_t& offset,
                         std::string& out) {
    if (offset >= text.size()) return false;
    const std::size_t start = offset;
    const unsigned char c = static_cast<unsigned char>(text[offset]);
    std::size_t width = 1;
    if (c < 0x80) {
        width = 1;
    } else if ((c & 0xE0) == 0xC0) {
        width = 2;
    } else if ((c & 0xF0) == 0xE0) {
        width = 3;
    } else if ((c & 0xF8) == 0xF0) {
        width = 4;
    } else {
        ++offset;
        out.assign("\xEF\xBF\xBD");
        return true;
    }
    if (offset + width > text.size()) {
        ++offset;
        out.assign("\xEF\xBF\xBD");
        return true;
    }
    out.assign(text.substr(start, width));
    offset += width;
    return true;
}

} // namespace

Tokenizer::Tokenizer() {
    vocabulary_.emplace("<pad>", pad());
    vocabulary_.emplace("<bos>", bos());
    vocabulary_.emplace("<eos>", eos());
    vocabulary_.emplace("<unk>", unk());
    vocabulary_.emplace("<s>", bos());
    vocabulary_.emplace("</s>", eos());
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
    piece_scores_.clear();

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto tab = line.find('\t');
        const std::string piece = tab == std::string::npos ? line : line.substr(0, tab);
        float score = 0.0f;
        if (tab != std::string::npos) {
            try {
                score = std::stof(line.substr(tab + 1));
            } catch (const std::exception&) {
                throw std::runtime_error("invalid tokenizer score in: " + vocab_path);
            }
        }
        const auto id = static_cast<TokenId>(pieces_.size());
        pieces_.push_back(piece);
        piece_scores_.push_back(score);
        piece_to_id_.emplace(piece, id);
    }
    if (pieces_.size() < 4)
        throw std::runtime_error("tokenizer vocab too small: " + vocab_path);
    if (piece_scores_.size() != pieces_.size())
        throw std::runtime_error("tokenizer score/piece count mismatch: " + vocab_path);
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
    if (text.empty()) return {};

    const std::string normalized = normalize_for_bpe(text);
    if (normalized.empty()) return {};

    // SentencePiece BPE starts at Unicode character granularity. Unknown
    // runs are represented by one UNK token rather than one UNK per Unicode
    // code point; this is important for text outside the learned vocabulary.
    std::vector<TokenId> tokens;
    std::size_t offset = 0;
    while (offset < normalized.size()) {
        std::string character;
        if (!next_utf8_codepoint(normalized, offset, character)) break;
        const auto it = piece_to_id_.find(character);
        if (it != piece_to_id_.end()) {
            tokens.push_back(it->second);
        } else if (tokens.empty() || tokens.back() != unk()) {
            tokens.push_back(unk());
        }
    }

    while (tokens.size() > 1) {
        float best_score = -1.0e30f;
        std::size_t best_idx = tokens.size();
        TokenId best_id = unk();

        for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
            const TokenId left = tokens[i];
            const TokenId right = tokens[i + 1];
            if (left >= pieces_.size() || right >= pieces_.size()) continue;
            if (left == unk() || right == unk()) continue;

            const std::string candidate = pieces_[left] + pieces_[right];
            const auto it = piece_to_id_.find(candidate);
            if (it == piece_to_id_.end()) continue;

            const TokenId merged = it->second;
            const float score = piece_scores_[merged];
            if (score > best_score) {
                best_score = score;
                best_idx = i;
                best_id = merged;
            }
        }

        if (best_idx == tokens.size()) break;
        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_idx + 1));
    }

    return tokens;
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
