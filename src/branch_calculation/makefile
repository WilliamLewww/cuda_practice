CUDA_PATH=/usr/local/cuda-10.1

NVCC=$(CUDA_PATH)/bin/nvcc
NVPROF=$(CUDA_PATH)/bin/nvprof

CUDA_FLAGS=--gpu-architecture=sm_35

CUDA_SRCS := $(wildcard *.cu)
CUDA_OBJS := $(CUDA_SRCS:%.cu=%.o)

all: clean $(CUDA_OBJS)
	$(NVCC) $(CUDA_FLAGS) *.o

%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) --device-c $^

clean:
	rm -rf *.o;
	rm -rf *.out;

profile:
	sudo $(NVPROF) ./*.out
