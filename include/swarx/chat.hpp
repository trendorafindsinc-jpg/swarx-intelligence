#pragma once

#include "swarx/inference.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace swarx {

enum class ChatRole {
    System,
    User,
    Assistant,
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::string content;
};

struct ChatConfig {
    InferenceConfig inference{};
    std::size_t max_history_messages = 32;
    bool include_assistant_reply_in_history = true;
};

struct ChatResult {
    ChatMessage message{ChatRole::Assistant, {}};
    InferenceResult inference{};
};

// High-level, dependency-free conversation boundary for SWARX. This layer
// turns role-aware messages into a deterministic prompt while leaving model
// execution, sampling, cancellation and token streaming to InferenceSession.
class ChatSession {
public:
    ChatSession();
    explicit ChatSession(TransformerModel model);
    explicit ChatSession(const std::string& weights_path);

    const InferenceSession& inference() const noexcept { return inference_; }
    InferenceSession& inference() noexcept { return inference_; }

    const std::vector<ChatMessage>& history() const noexcept { return history_; }
    void clear() noexcept;
    void set_system_prompt(std::string prompt);
    const std::string& system_prompt() const noexcept { return system_prompt_; }

    ChatResult send(std::string_view user_message,
                    ChatConfig config = {},
                    TokenCallback on_token = {},
                    CancelCallback should_cancel = {});

    ChatResult send(const ChatMessage& user_message,
                    ChatConfig config = {},
                    TokenCallback on_token = {},
                    CancelCallback should_cancel = {});

private:
    std::string build_prompt(const ChatConfig& config) const;
    void trim_history(std::size_t max_messages);

    InferenceSession inference_;
    std::string system_prompt_;
    std::vector<ChatMessage> history_;
};

} // namespace swarx
