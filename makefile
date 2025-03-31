CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++11 -Wall -Wextra -O2
NVCCFLAGS := -std=c++11 -arch=sm_50 -O2

INCLUDES := inc/knn_cuda.h

SRCS := main.cpp src/knn_cuda.cpp
OBJS := $(SRCS:.cpp=.o)


TARGET := MyProject

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean