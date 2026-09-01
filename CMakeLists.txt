cmake_minimum_required(VERSION 3.20)
project(swarx_intelligence VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SWARX_BUILD_TESTS "Build SWARX Intelligence tests" ON)
option(SWARX_BUILD_CLI "Build the swarx-cli executable" ON)

add_library(swarx_core
    src/core/tensor.cpp
    src/core/tokenizer.cpp
    src/core/sampler.cpp
    src/runtime/engine.cpp
)

target_include_directories(swarx_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(swarx_core PUBLIC cxx_std_20)

if(SWARX_BUILD_CLI)
    add_executable(swarx-cli src/main.cpp)
    target_link_libraries(swarx-cli PRIVATE swarx_core)
endif()

if(SWARX_BUILD_TESTS)
    enable_testing()
    add_executable(swarx-smoke-tests tests/smoke.cpp)
    target_link_libraries(swarx-smoke-tests PRIVATE swarx_core)
    add_test(NAME swarx-smoke-tests COMMAND swarx-smoke-tests)
endif()
