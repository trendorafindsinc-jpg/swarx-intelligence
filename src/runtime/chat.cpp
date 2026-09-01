#include "swarx/chat.hpp"

#include <stdexcept>
#include <utility>

namespace swarx {
namespace {

const char* role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System: return "system";
    case ChatRole::User: return "user";
    case ChatRole::Assistant: return "assistant";
    }
    return "user";
}

} // namespace

ChatSession::ChatSession() = default;

ChatSession::ChatSession(TransformerModel model)
    : inference_(std::move(model)) {}

ChatSession::ChatSession(const std::string& weights_path)
    : inference_(weights_path) {}

void ChatSession::clear() noexcept {
    history_.clear();
    inference_.reset();
}

void ChatSession::set_system_prompt(std::string prompt) {
    system_prompt_ = std::move(prompt);
    inference_.reset();
}

std::string ChatSession::build_prompt(const ChatConfig& config) const {
    std::string prompt;
    prompt.reserve(system_prompt_.size() + history_.size() * 32 + 64);

    if (!system_prompt_.empty()) {
        prompt += "<|system|>\n";
        prompt += system_prompt_;
        prompt += "\n";
    }

    const std::size_t begin = history_.size() > config.max_history_messages
        ? history_.size() - config.max_history_messages : 0;
    for (std::size_t i = begin; i < history_.size(); ++i) {
        prompt += "<|";
        prompt += role_name(history_[i].role);
        prompt += "|>\n";
        prompt += history_[i].content;
        prompt += "\n";
    }

    prompt += "<|assistant|>\n";
    return prompt;
}

void ChatSession::trim_history(std::size_t max_messages) {
    if (max_messages == 0) {
        history_.clear();
        return;
    }
    if (history_.size() > max_messages)
        history_.erase(history_.begin(), history_.end() - static_cast<std::ptrdiff_t>(max_messages));
}

ChatResult ChatSession::send(std::string_view user_message,
                             ChatConfig config,
                             TokenCallback on_token,
                             CancelCallback should_cancel) {
    return send(ChatMessage{ChatRole::User, std::string(user_message)},
                std::move(config), std::move(on_token), std::move(should_cancel));
}

ChatResult ChatSession::send(const ChatMessage& user_message,
                             ChatConfig config,
                             TokenCallback on_token,
                             CancelCallback should_cancel) {
    if (user_message.role != ChatRole::User)
        throw std::invalid_argument("SWARX ChatSession::send expects a user message");
    if (user_message.content.empty())
        throw std::invalid_argument("SWARX chat message cannot be empty");

    history_.push_back(user_message);
    trim_history(config.max_history_messages);

    const std::string prompt = build_prompt(config);
    auto inference_result = inference_.generate(prompt, config.inference,
                                                std::move(on_token),
                                                std::move(should_cancel));

    ChatResult result;
    result.message = ChatMessage{ChatRole::Assistant, inference_result.text};
    result.inference = std::move(inference_result);

    if (config.include_assistant_reply_in_history)
        history_.push_back(result.message);
    trim_history(config.max_history_messages);
    return result;
}

} // namespace swarx
