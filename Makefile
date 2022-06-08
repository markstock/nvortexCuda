NVCC=nvcc
ARCH=--gpu-architecture=compute_86 --gpu-code=sm_86
NVCCFLAGS=-O2 --use_fast_math -lineinfo
CXXFLAGS=-march=native,-fopenmp

all : nvCuda01.bin nvCuda02.bin nvCuda03.bin

%.bin : %.cu
	$(NVCC) $(ARCH) $(NVCCFLAGS) -Xcompiler "$(CXXFLAGS)" -o $@ $^
