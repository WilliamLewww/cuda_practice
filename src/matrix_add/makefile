CUDA_PATH=/usr/local/cuda-10.1

NVCC=$(CUDA_PATH)/bin/nvcc

CUDA_FLAGS=--gpu-architecture=sm_35

CUDA_SRCS := $(wildcard *.cu)
CUDA_OBJS := $(CUDA_SRCS:%.cu=%.o)

all: $(CUDA_OBJS)
	$(NVCC) $(CUDA_FLAGS) *.o

%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) --device-c $^