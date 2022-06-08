# nvortexCuda
Simple n-body program to test Cuda performance

## Compile and run
As long CUDA is installed and `nvcc` is in your PATH, you should be able to do:

    make
    ./nvCuda01.bin
    ./nvCuda02.bin
    ./nvCuda03.bin

## Description
This repository contains a few progressive examples of a compute-only n-body calculation
of the Biot-Savart influence of N desingularized vorticies on one another.

Program `nvCuda01.cu` is the simplest implementation. On the CPU side, the program parallelizes
with a basic OpenMP `parallel for` loop over the target particles. On the GPU side, we use CUDA
without unified or pinned memory (full transfers), with one target particle per "thread."

Program `nvCuda02.cu` speeds this up considerably. The CPU now uses `omp simd` to vectorize the
inner loop over source particles. The GPU uses shared memory to load blocks of source particles
in a coalesced manner before all threads operate on that block. This program represents the
"80" part of the "80-20 rule": that you can go most of the way with some simple methods.

Program `nvCuda03` adds some enhancements in an attempt to eke out even more performance, though
only on the GPU side.

## Other codes
If you want to see how other libraries and methodologies improve performance on this problem,
look at some of my other repositories:

* [nvortexVc](https://github.com/Applied-Scientific-Research/nvortexVc) - using [Vc](https://github.com/VcDevel/Vc) for explicit vectorization
* [onbody](https://github.com/Applied-Scientific-Research/onbody) - using CPU treecodes with better order of operations
* [Omega2D](https://github.com/Applied-Scientific-Research/Omega2D) - a complete 2D vortex methods simulator
