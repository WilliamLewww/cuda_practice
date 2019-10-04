CUDAPATH=/usr/local/cuda-10.1

CC=g++
NVCC=$(CUDAPATH)/bin/nvcc
NVPROF=$(CUDAPATH)/bin/nvprof
MEMCHECK=$(CUDAPATH)/bin/cuda-memcheck
NSIGHTCLI=$(CUDAPATH)/bin/nv-nsight-cu-cli
CUDAFLAGS=--gpu-architecture=sm_50
CURRENTFILE=lab_1_2

all: clean compile run

clean:
	rm -rf *.out

compile:
	$(NVCC) $(CUDAFLAGS) ./$(CURRENTFILE).cu -o ./$(CURRENTFILE).out

run:
	./$(CURRENTFILE).out