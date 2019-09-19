CC=g++
NVCC=/usr/local/cuda-10.1/bin/nvcc
NVPROF = /usr/local/cuda-10.1/bin/nvprof
CXXFLAGS=
CUDAFLAGS=-m64 -gencode arch=compute_30,code=compute_30
LIBS=
LIBDIRS=
INCDIRS=

all: clean compile

clean:
	rm -rf bin

compile:
	mkdir -p bin
	$(NVCC) $(CUDAFLAGS) ./src/matrix_add.cu -o ./bin/matrix_add.out

profile:
	sudo $(NVPROF) ./bin/matrix_add.out