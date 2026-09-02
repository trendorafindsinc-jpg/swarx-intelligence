#include "swarx/engine.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string model_path;
    std::string vocab_path;
    std::string tokenize_text;
    std::string logits_text;
    std::string generate_text;
    std::size_t max_new_tokens = 0;
    bool tokenize_mode = false;
    bool logits_mode = false;
    bool generate_mode = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "--vocab" || arg == "-v") && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (arg == "--tokenize" && i + 1 < argc) {
            tokenize_mode = true;
            tokenize_text = argv[++i];
        } else if (arg == "--logits" && i + 1 < argc) {
            logits_mode = true;
            logits_text = argv[++i];
        } else if (arg == "--generate" && i + 1 < argc) {
            generate_mode = true;
            generate_text = argv[++i];
        } else if (arg == "--max-new" && i + 1 < argc) {
            max_new_tokens = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: swarx-cli [--model path/to/model.swarx] [--vocab path/to/file.vocab]\n"
                         "                  [--tokenize \"text\" --vocab path/to/file.vocab]\n"
                         "                  [--logits \"text\" --model path/to/model.swarx --vocab path/to/file.vocab]\n"
                         "                  [--generate \"text\" --max-new N --model path/to/model.swarx --vocab path/to/file.vocab]\n";
            return 0;
        }
    }

    swarx::Engine engine;
    if (!vocab_path.empty()) {
        engine.load_tokenizer_vocab(vocab_path);
        if (!tokenize_mode && !logits_mode && !generate_mode) {
            std::cout << "Loaded tokenizer vocab: " << vocab_path << "\n";
        }
    }

    if (tokenize_mode) {
        if (vocab_path.empty()) {
            std::cerr << "error: --tokenize requires --vocab\n";
            return 2;
        }
        const auto ids = engine.tokenizer().encode(tokenize_text, true, false);
        std::cout << "TOKENS:";
        for (const auto id : ids) std::cout << ' ' << id;
        std::cout << '\n';
        return 0;
    }

    if (!model_path.empty()) {
        engine.load_model(model_path);
        if (!logits_mode && !generate_mode) {
            std::cout << "Loaded SWARX model: " << model_path << "\n";
        }
    }

    if (logits_mode) {
        if (model_path.empty() || vocab_path.empty()) {
            std::cerr << "error: --logits requires --model and --vocab\n";
            return 2;
        }
        const auto logits = engine.next_token_logits(logits_text);
        std::cout << "LOGITS:";
        for (const float value : logits) std::cout << ' ' << value;
        std::cout << '\n';
        return 0;
    }

    if (generate_mode) {
        if (model_path.empty() || vocab_path.empty()) {
            std::cerr << "error: --generate requires --model and --vocab\n";
            return 2;
        }
        std::cout << "GENERATED: " << engine.generate(generate_text, max_new_tokens) << '\n';
        return 0;
    }

    std::cout << "SWARX Intelligence\n";
    std::cout << (engine.has_model()
                      ? "Model backend active. Type a prompt, or 'exit' to quit.\n\n"
                      : "Bootstrap mode (no model). Type a prompt, or 'exit' to quit.\n\n");

    std::string prompt;
    while (std::cout << "> " && std::getline(std::cin, prompt)) {
        if (prompt == "exit" || prompt == "quit") break;
        if (prompt.empty()) continue;
        try {
            std::cout << engine.generate(prompt) << "\n\n";
        } catch (const std::exception& ex) {
            std::cerr << "error: " << ex.what() << "\n\n";
        }
    }

    return 0;
}
