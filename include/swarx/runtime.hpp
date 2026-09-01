#pragma once

#include "swarx/chat.hpp"
#include "swarx/model_identity.hpp"
#include "swarx/model_inspector.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace swarx {

struct RuntimeConfig {
    std::string system_prompt;
    ChatConfig chat{};
};

struct RuntimeHealth {
    bool model_loaded = false;
    bool model_valid = false;
    bool finite_parameters = false;
    std::size_t parameter_count = 0;
    std::size_t parameter_bytes = 0;
    std::size_t vocabulary_size = 0;
    std::size_t context_length = 0;
    std::string model_fingerprint;
    std::string algorithm;
};

// Stable product-facing runtime boundary for SWARX. Applications should depend
// on this facade instead of reaching into individual engine subsystems.
class Runtime {
public:
    Runtime();
    explicit Runtime(RuntimeConfig config);
    explicit Runtime(TransformerModel model, RuntimeConfig config = {});
    explicit Runtime(const std::string& weights_path, RuntimeConfig config = {});

    bool loaded() const noexcept;
    const RuntimeConfig& config() const noexcept { return config_; }
    const RuntimeHealth& health() const noexcept { return health_; }
    const ModelIdentity& identity() const noexcept { return identity_; }
    const ModelInspection& inspection() const noexcept { return inspection_; }

    void configure(RuntimeConfig config);
    void load(const std::string& weights_path);
    void save(const std::string& weights_path) const;
    void reset() noexcept;
    void validate() const;
    void refresh_health();

    ChatResult chat(std::string_view user_message,
                    TokenCallback on_token = {},
                    CancelCallback should_cancel = {});

    ChatSession& chat_session() noexcept { return chat_; }
    const ChatSession& chat_session() const noexcept { return chat_; }

private:
    void rebuild_state();

    RuntimeConfig config_{};
    ChatSession chat_{};
    RuntimeHealth health_{};
    ModelIdentity identity_{};
    ModelInspection inspection_{};
};

} // namespace swarx
