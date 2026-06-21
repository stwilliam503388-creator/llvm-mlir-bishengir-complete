CXX = /usr/bin/c++
CXXFLAGS = -std=gnu++17 -arch arm64 -O3
LLVM = /opt/homebrew/opt/llvm
INCS = -I$(LLVM)/include -Iinclude
LIBS = $(LLVM)/lib/libMLIR.dylib
SRCS = tools/standalone-opt.cpp
OBJS = standalone-opt.o
TARGET = standalone-opt

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

$(OBJS): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET) test/example.mlir

.PHONY: all clean run
