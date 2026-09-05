cmake_minimum_required(VERSION 3.20)
project(swarx_intelligence VERSION 0.24.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(SWARX_BUILD_TESTS "Build SWARX Intelligence tests" ON)
option(SWARX_BUILD_CLI "Build the swarx-cli executable" ON)

add_library(swarx_core
    src/core/tensor.cpp src/core/tensor_ops.cpp src/core/tokenizer.cpp src/core/sampler.cpp
    src/runtime/engine.cpp src/runtime/transformer.cpp src/runtime/attention.cpp src/runtime/model.cpp
    src/runtime/generation.cpp src/runtime/weights.cpp src/runtime/kv_cache.cpp src/runtime/inference.cpp
    src/runtime/model_inspector.cpp src/runtime/engine_pipeline.cpp src/runtime/model_identity.cpp
    src/runtime/chat.cpp src/runtime/runtime.cpp src/runtime/backend.cpp src/runtime/quantization.cpp
    src/runtime/serving.cpp src/runtime/web.cpp src/runtime/online.cpp src/runtime/source_trust.cpp
    src/runtime/intelligence_router.cpp src/runtime/tool_registry.cpp src/runtime/intelligence_bus.cpp
    src/runtime/knowledge.cpp src/runtime/online_orchestrator.cpp src/runtime/evidence.cpp
    src/runtime/research.cpp src/runtime/source_registry.cpp src/runtime/source_adapters.cpp
    src/runtime/resilience_gateway.cpp
)
target_include_directories(swarx_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(swarx_core PUBLIC cxx_std_20)
target_compile_options(swarx_core PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>)

if(SWARX_BUILD_CLI)
    add_executable(swarx-cli src/main.cpp)
    target_link_libraries(swarx-cli PRIVATE swarx_core)
endif()

if(SWARX_BUILD_TESTS)
    enable_testing()
    add_executable(swarx-smoke-tests tests/smoke.cpp)
    target_link_libraries(swarx-smoke-tests PRIVATE swarx_core)
    add_test(NAME swarx-smoke-tests COMMAND swarx-smoke-tests)
    add_executable(swarx-backend-tests tests/backend_smoke.cpp)
    target_link_libraries(swarx-backend-tests PRIVATE swarx_core)
    add_test(NAME swarx-backend-tests COMMAND swarx-backend-tests)
    add_executable(swarx-serving-tests tests/serving_smoke.cpp)
    target_link_libraries(swarx-serving-tests PRIVATE swarx_core)
    add_test(NAME swarx-serving-tests COMMAND swarx-serving-tests)
    add_executable(swarx-online-tests tests/online_smoke.cpp)
    target_link_libraries(swarx-online-tests PRIVATE swarx_core)
    add_test(NAME swarx-online-tests COMMAND swarx-online-tests)
    add_executable(swarx-intelligence-tests tests/intelligence_router_smoke.cpp)
    target_link_libraries(swarx-intelligence-tests PRIVATE swarx_core)
    add_test(NAME swarx-intelligence-tests COMMAND swarx-intelligence-tests)
    add_executable(swarx-intelligence-bus-tests tests/intelligence_bus_smoke.cpp)
    target_link_libraries(swarx-intelligence-bus-tests PRIVATE swarx_core)
    add_test(NAME swarx-intelligence-bus-tests COMMAND swarx-intelligence-bus-tests)
    add_executable(swarx-knowledge-tests tests/knowledge_smoke.cpp)
    target_link_libraries(swarx-knowledge-tests PRIVATE swarx_core)
    add_test(NAME swarx-knowledge-tests COMMAND swarx-knowledge-tests)
    add_executable(swarx-source-trust-tests tests/source_trust_smoke.cpp)
    target_link_libraries(swarx-source-trust-tests PRIVATE swarx_core)
    add_test(NAME swarx-source-trust-tests COMMAND swarx-source-trust-tests)
    add_executable(swarx-online-orchestrator-tests tests/online_orchestrator_smoke.cpp)
    target_link_libraries(swarx-online-orchestrator-tests PRIVATE swarx_core)
    add_test(NAME swarx-online-orchestrator-tests COMMAND swarx-online-orchestrator-tests)
    add_executable(swarx-evidence-tests tests/evidence_smoke.cpp)
    target_link_libraries(swarx-evidence-tests PRIVATE swarx_core)
    add_test(NAME swarx-evidence-tests COMMAND swarx-evidence-tests)
    add_executable(swarx-research-tests tests/research_smoke.cpp)
    target_link_libraries(swarx-research-tests PRIVATE swarx_core)
    add_test(NAME swarx-research-tests COMMAND swarx-research-tests)
    add_executable(swarx-source-registry-tests tests/source_registry_smoke.cpp)
    target_link_libraries(swarx-source-registry-tests PRIVATE swarx_core)
    add_test(NAME swarx-source-registry-tests COMMAND swarx-source-registry-tests)
    add_executable(swarx-source-adapter-tests tests/source_adapters_smoke.cpp)
    target_link_libraries(swarx-source-adapter-tests PRIVATE swarx_core)
    add_test(NAME swarx-source-adapter-tests COMMAND swarx-source-adapter-tests)
    add_executable(swarx-resilience-gateway-tests tests/resilience_gateway_smoke.cpp)
    target_link_libraries(swarx-resilience-gateway-tests PRIVATE swarx_core)
    add_test(NAME swarx-resilience-gateway-tests COMMAND swarx-resilience-gateway-tests)
endif()
