#include "swarx/runtime.hpp"

#include <stdexcept>
#include <utility>

namespace swarx {

Runtime::Runtime() = default;

Runtime::Runtime(RuntimeConfig config)
    : config_(std::move(config)) {
    chat_.set_system_prompt(config_.system_prompt);
    refresh_health();
}

Runtime::Runtime(TransformerModel model, RuntimeConfig config)
    : config_(std::move(config)), chat_(std::move(model)) {
    chat_.set_system_prompt(config_.system_prompt);
    refresh_health();
}

Runtime::Runtime(const std::string& weights_path, RuntimeConfig config)
    : config_(std::move(config)), chat_(weights_path) {
    chat_.set_system_prompt(config_.system_prompt);
    refresh_health();
}

bool Runtime::loaded() const noexcept {
    return health_.model_loaded;
}

void Runtime::configure(RuntimeConfig config) {
    config_ = std::move(config);
    chat_.set_system_prompt(config_.system_prompt);
}

void Runtime::load(const std::string& weights_path) {
    chat_.inference().load_weights(weights_path);
    rebuild_state();
}

void Runtime::save(const std::string& weights_path) const {
    if (!loaded()) throw std::logic_error("SWARX runtime has no loaded model");
    chat_.inference().save_weights(weights_path);
}

void Runtime::reset() noexcept {
    chat_.clear();
}

void Runtime::validate() const {
    if (!loaded()) throw std::logic_error("SWARX runtime has no loaded model");
    ModelInspector::validate(chat_.inference().model());
}

void Runtime::refresh_health() {
    health_ = {};
    if (chat_.inference().model_config().vocab_size == 0) return;

    const auto& model = chat_.inference().model();
    inspection_ = ModelInspector::inspect(model);
    identity_ = ModelIdentityProvider::identify(model);

    health_.model_loaded = true;
    health_.model_valid = inspection_.warnings.empty();
    health_.finite_parameters = inspection_.finite;
    health_.parameter_count = inspection_.parameter_count;
    health_.parameter_bytes = inspection_.parameter_bytes;
    health_.vocabulary_size = inspection_.vocab_size;
    health_.context_length = inspection_.max_sequence_length;
    health_.model_fingerprint = identity_.fingerprint;
    health_.algorithm = identity_.algorithm;
}

void Runtime::rebuild_state() {
    chat_.clear();
    chat_.set_system_prompt(config_.system_prompt);
    refresh_health();
}

ChatResult Runtime::chat(std::string_view user_message,
                         TokenCallback on_token,
                         CancelCallback should_cancel) {
    if (!loaded()) throw std::logic_error("SWARX runtime has no loaded model");
    return chat_.send(user_message, config_.chat,
                      std::move(on_token), std::move(should_cancel));
}

} // namespace swarx
