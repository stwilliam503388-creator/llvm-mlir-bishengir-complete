cmake_minimum_required(VERSION 3.20)

# 使用 Homebrew LLVM 的 clang++（必须在 project() 之前设置）
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

project(HelloPass)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_library(HelloPass MODULE HelloPass.cpp)

# 链接 LLVM 共享库（LLVM_LIBRARY_DIR 是单数）
target_link_libraries(HelloPass PRIVATE ${LLVM_LIBRARY_DIR}/libLLVM${CMAKE_SHARED_LIBRARY_SUFFIX})
