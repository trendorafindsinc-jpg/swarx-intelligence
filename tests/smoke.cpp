#include "swarx/engine.hpp"
#include "swarx/sampler.hpp"
#include "swarx/tensor.hpp"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    swarx::Tensor tensor({2, 3});
    assert(tensor.size() == 6);

    swarx::Sampler sampler(1234);
    assert(sampler.greedy({-1.0f, 3.0f, 2.0f}) == 1);

    swarx::Engine engine;
    const auto tokens = engine.tokenize("hello world");
    assert(tokens.size() == 2);
    assert(engine.generate("hello").find("SWARX Intelligence runtime is ready") != std::string::npos);

    return 0;
}
