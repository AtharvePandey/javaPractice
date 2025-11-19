CXX = g++
CXXFLAGS = -std=c++23 -O2 -Wall -Wextra

SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)

# Programs are the .cpp files containing main()
PROGRAM_SRCS := $(shell grep -l "int main" $(SRCS))
PROGRAMS := $(PROGRAM_SRCS:.cpp=)

all: $(PROGRAMS)

# Each program links ALL object files
$(PROGRAMS): %: %.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

# Compile rule
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(PROGRAMS) *.o
