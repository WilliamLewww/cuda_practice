CC=g++
NVCC=/usr/local/cuda-10.1/bin/nvcc
NVPROF=/usr/local/cuda-10.1/bin/nvprof
MEMCHECK=/usr/local/cuda-10.1/bin/cuda-memcheck
CXXFLAGS=
CUDAFLAGS=--gpu-architecture=sm_50
LIBS=
LIBDIRS=
INCDIRS=

all: clean compile run

clean:
	rm -rf bin

compile:
	mkdir -p bin
	$(NVCC) $(CUDAFLAGS) ./src/array_sum.cu -o ./bin/array_sum.out
run:
	./bin/array_sum.out

memory-check:
	$(MEMCHECK) ./bin/array_sum.out

profile:
	mkdir -p dump
	cd dump; sudo $(NVPROF) ../bin/array_sum.out 2>profile.log; cat profile.log;

profile-metrics:
	mkdir -p dump
	cd dump; sudo $(NVPROF) --metrics all ../bin/array_sum.out 2>profile-metrics.log; cat profile-metrics.log;

profile-events:
	mkdir -p dump
	cd dump; sudo $(NVPROF) --events all ../bin/array_sum.out 2>profile-events.log; cat profile-events.log;