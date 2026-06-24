cmake_minimum_required(VERSION 3.20)
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
project(HelloMLIRPass)
set(CMAKE_CXX_STANDARD 17)

include_directories("/opt/homebrew/Cellar/llvm/22.1.6/include")

set(LLVM_DIR "/opt/homebrew/Cellar/llvm/22.1.6/lib/cmake/llvm")
set(MLIR_DIR "/opt/homebrew/Cellar/llvm/22.1.6/lib/cmake/mlir")
find_package(MLIR REQUIRED CONFIG)

add_executable(hello-mlir hello-mlir.cpp)

set(MLIR_LIBS
  MLIRFuncDialect MLIRIR MLIRPass MLIRSupport
  MLIRParser MLIRTransformUtils MLIRAnalysis
)
target_link_libraries(hello-mlir PRIVATE ${MLIR_LIBS})
