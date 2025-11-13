# Makefile for RTM stencil lab

NVCC      := nvcc
NVCCFLAGS := -O3 -std=c++14 -arch=sm_86

TARGET    := rtm_lab
SRC_CU    := main.cu
SRC_CPP   := rtm_cpu.cpp utils.cpp

OBJ       := $(SRC_CU:.cu=.o) $(SRC_CPP:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
