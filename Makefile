#CUDA_BIN=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/compilers/bin
#NVCC=$(CUDA_BIN)/nvcc
NVCC=nvcc
ARCH=--gpu-architecture=compute_80 --gpu-code=sm_80
NVCCFLAGS=-O2 --use_fast_math -lineinfo
#NVCCFLAGS=-O2 -lineinfo
CXXFLAGS=-march=native,-fopenmp,-fomit-frame-pointer

all : nvCuda01.bin nvCuda02.bin nvCuda03.bin nvCuda04.bin

%.bin : %.cu
	$(NVCC) $(ARCH) $(NVCCFLAGS) -Xcompiler "$(CXXFLAGS)" -o $@ $^

clean :
	rm nvCuda01.bin nvCuda02.bin nvCuda03.bin nvCuda04.bin
