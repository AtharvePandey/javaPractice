CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

# Find all .cpp files and create a target for each (L1.cpp -> L1, L2.cpp -> L2, ...)
ALL_SRCS := $(wildcard *.cpp)

# Detect which .cpp files contain a `main` function and only build those as programs.
# This prevents linker errors when there are utility/source files without `main`.
# `PROGRAM_SRCS` will be a list of .cpp files that include `main`.
ifeq ($(ALL_SRCS),)
PROGRAM_SRCS :=
else
PROGRAM_SRCS := $(shell grep -l "int main" $(ALL_SRCS) 2>/dev/null || true)
endif
TARGETS := $(PROGRAM_SRCS:.cpp=)

.PHONY: all clean run

ifeq ($(TARGETS),)
all:
	@echo "No programs with main found (no executables to build)."
else
all: $(TARGETS)
endif

# Compile each .cpp into a .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link rule: build executable from object file(s). For single-file programs this links the .o into the executable.
# If your program has multiple source files, create a separate target or adjust dependencies accordingly.
%: %.o
	$(CXX) $(CXXFLAGS) -o $@ $^

# Run: execute the first built target (convenience)
run:
	@if [ -z "$(TARGETS)" ]; then \
		echo "No targets to run"; \
	else \
		./$(firstword $(TARGETS)); \
	fi

clean:
	rm -f $(TARGETS) *.o