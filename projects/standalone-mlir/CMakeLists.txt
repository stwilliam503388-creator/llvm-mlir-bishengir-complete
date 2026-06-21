cmake_minimum_required(VERSION 3.20)
project(StandaloneMLIR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 手动配置 MLIR 路径（跳过 find_package，避免 AddMLIR.cmake 冲突）
set(LLVM_HOME "/opt/homebrew/opt/llvm")
set(MLIR_INC "${LLVM_HOME}/include")
set(MLIR_LIB "${LLVM_HOME}/lib/libMLIR.dylib")

include_directories(${MLIR_INC})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(standalone-opt
    tools/standalone-opt.cpp
)

target_link_libraries(standalone-opt "${MLIR_LIB}")
