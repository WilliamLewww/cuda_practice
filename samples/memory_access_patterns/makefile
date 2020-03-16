CUDA_PATH=/usr/local/cuda-10.2

CURRENT_PATH=$(shell pwd)

CC=g++
NVCC=$(CUDA_PATH)/bin/nvcc
NVPROF=$(CUDA_PATH)/bin/nvprof
NSIGHT_CLI=$(CUDA_PATH)/bin/nv-nsight-cu-cli
NVVP=$(CUDA_PATH)/bin/nvvp
GDB=gdb
CUDA_GDB=$(CUDA_PATH)/bin/cuda-gdb
MEMCHECK=$(CUDA_PATH)/bin/cuda-memcheck

CUDA_FLAGS=--gpu-architecture=sm_35

CUDA_SRCS := $(wildcard *.cu)
CUDA_OBJS := $(CUDA_SRCS:%.cu=%.o)

all: $(CUDA_OBJS)
	$(NVCC) $(CUDA_FLAGS) *.o

%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) --device-c $^