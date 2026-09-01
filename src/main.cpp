#include "swarx/engine.hpp"

#include <iostream>
#include <string>

int main() {
    swarx::Engine engine;

    std::cout << "SWARX Intelligence v0.1.0\n";
    std::cout << "Type a prompt, or 'exit' to quit.\n\n";

    std::string prompt;
    while (std::cout << "> " && std::getline(std::cin, prompt)) {
        if (prompt == "exit" || prompt == "quit") break;
        if (prompt.empty()) continue;
        std::cout << engine.generate(prompt) << "\n\n";
    }

    return 0;
}
