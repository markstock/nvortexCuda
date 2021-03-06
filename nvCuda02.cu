/*
 * nvortexCuda.cpp
 *
 * (c)2022 Mark J. Stock <markjstock@gmail.com>
 *
 * v0.2  add shared memory
 */

#include <vector>
#include <random>
#include <chrono>

#include <cuda_runtime.h>


// compute using float or double
#define FLOAT float

// threads per block (hard coded)
#define THREADS_PER_BLOCK 128

// -------------------------
// compute kernel - GPU
__global__ void nvortex_2d_nograds_gpu(
    const int32_t nSrc,
    const FLOAT* const sx,
    const FLOAT* const sy,
    const FLOAT* const ss,
    const FLOAT* const sr,
    const int32_t tOffset,
    const FLOAT* const tx,
    const FLOAT* const ty,
    const FLOAT* const tr,
    FLOAT* const tu,
    FLOAT* const tv) {

  // local "thread" id - this is the target particle
  const int32_t i = tOffset + blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;

  // load sources into shared memory (or not)
  __shared__ FLOAT s_sx[THREADS_PER_BLOCK];
  __shared__ FLOAT s_sy[THREADS_PER_BLOCK];
  __shared__ FLOAT s_ss[THREADS_PER_BLOCK];
  __shared__ FLOAT s_sr[THREADS_PER_BLOCK];

  // velocity accumulators for target point
  FLOAT locu = 0.0f;
  FLOAT locv = 0.0f;

  for (int32_t b=0; b<nSrc/THREADS_PER_BLOCK; ++b) {

    const int32_t gidx = b*THREADS_PER_BLOCK + threadIdx.x;
    s_sx[threadIdx.x] = sx[gidx];
    s_sy[threadIdx.x] = sy[gidx];
    s_ss[threadIdx.x] = ss[gidx];
    s_sr[threadIdx.x] = sr[gidx];
    __syncthreads();

    // loop over all source points
    for (int32_t j=0; j<THREADS_PER_BLOCK; ++j) {
      FLOAT dx = s_sx[j] - tx[i];
      FLOAT dy = s_sy[j] - ty[i];
      FLOAT distsq = dx*dx + dy*dy + s_sr[j]*s_sr[j] + tr[i]*tr[i];
      FLOAT factor = s_ss[j] / distsq;
      locu += dy * factor;
      locv -= dx * factor;
    }

    __syncthreads();
  }

  // save into device view
  // use atomics?!?
  tu[i] = locu / (2.0f*3.1415926536f);
  tv[i] = locv / (2.0f*3.1415926536f);

  return;
}

// -------------------------
// compute kernel - CPU
__host__ void nvortex_2d_nograds_cpu(
    const int32_t nSrc,
    const FLOAT* const sx,
    const FLOAT* const sy,
    const FLOAT* const ss,
    const FLOAT* const sr,
    const FLOAT tx,
    const FLOAT ty,
    const FLOAT tr,
    FLOAT* const tu,
    FLOAT* const tv) {

  // velocity accumulators for target point
  FLOAT locu = 0.0f;
  FLOAT locv = 0.0f;

  // loop over all source points
  #pragma omp simd reduction(+:locu,locv)
  for (int32_t j=0; j<nSrc; ++j) {
    FLOAT dx = sx[j] - tx;
    FLOAT dy = sy[j] - ty;
    FLOAT distsq = dx*dx + dy*dy + sr[j]*sr[j] + tr*tr;
    FLOAT factor = ss[j] / distsq;
    locu += dy * factor;
    locv -= dx * factor;
  }

  // save into device view
  // use atomics?!?
  *tu = locu / (2.0f*3.1415926536f);
  *tv = locv / (2.0f*3.1415926536f);

  return;
}

// not really alignment, just minimum block sizes
__host__ int32_t buffer(const int32_t _n, const int32_t _align) {
  // 63,64 returns 1; 64,64 returns 1; 65,64 returns 2
  return _align*(1+(_n-1)/_align);
}

// main program

static void usage() {
  fprintf(stderr, "Usage: nvCuda02 [-n=<number>]\n");
  exit(1);
}

int main(int argc, char **argv) {

  // number of particles/points
  int32_t npart = 200000;

  if (argc > 1) {
    if (strncmp(argv[1], "-n=", 3) == 0) {
      int num = atoi(argv[1] + 3);
      if (num < 1) usage();
      npart = num;
    }
  }

  printf( "performing 2D vortex Biot-Savart on %d points\n", npart);

  // number of GPUs present
  const int32_t ngpus = 1;
  // number of cuda streams to break work into
  const int32_t nstreams = 1;
  printf( "  ngpus ( %d )  and nstreams ( %d )\n", ngpus, nstreams);

  // set stream sizes
  const int32_t nperstrm = buffer(npart/nstreams, THREADS_PER_BLOCK);
  const int32_t npfull = nstreams*nperstrm;
  printf( "  nperstrm ( %d )  and npfull ( %d )\n", nperstrm, npfull);

  // define the host arrays (for now, sources and targets are the same)
  std::vector<FLOAT> hsx(npfull), hsy(npfull), hss(npfull), hsr(npfull), htu(npfull), htv(npfull);
  const FLOAT thisstrmag = 1.0 / std::sqrt(npart);
  const FLOAT thisrad    = (2./3.) / std::sqrt(npart);
  //std::random_device dev;
  //std::mt19937 rng(dev());
  std::mt19937 rng(1234);
  std::uniform_real_distribution<FLOAT> xrand(0.0,1.0);
  for (int32_t i = 0; i < npart; ++i)      hsx[i] = xrand(rng);
  for (int32_t i = npart; i < npfull; ++i) hsx[i] = 0.0;
  for (int32_t i = 0; i < npart; ++i)      hsy[i] = xrand(rng);
  for (int32_t i = npart; i < npfull; ++i) hsy[i] = 0.0;
  for (int32_t i = 0; i < npart; ++i)      hss[i] = thisstrmag * (2.0*xrand(rng)-1.0);
  for (int32_t i = npart; i < npfull; ++i) hss[i] = 0.0;
  for (int32_t i = 0; i < npart; ++i)      hsr[i] = thisrad;
  for (int32_t i = npart; i < npfull; ++i) hsr[i] = thisrad;
  for (int32_t i = 0; i < npfull; ++i)     htu[i] = 0.0;
  for (int32_t i = 0; i < npfull; ++i)     htv[i] = 0.0;

  // -------------------------
  // do a CPU version

  auto start = std::chrono::system_clock::now();

  #pragma omp parallel for
  for (int32_t i=0; i<npart; ++i) {
    nvortex_2d_nograds_cpu(npart, hsx.data(),hsy.data(),hss.data(),hsr.data(), hsx[i],hsy[i],hsr[i], &htu[i],&htv[i]);
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  double time = elapsed_seconds.count();

  printf( "  host total time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(4+14*(double)npart)/time);
  printf( "    results ( %g %g %g %g %g %g)\n", htu[0], htv[0], htu[1], htv[1], htu[npart-1], htv[npart-1]);

  // copy the results into temp vectors
  std::vector<FLOAT> htu_cpu(htu);
  std::vector<FLOAT> htv_cpu(htv);

  // -------------------------
  // do the GPU version

  // set device pointers, too
  FLOAT *dsx, *dsy, *dss, *dsr;
  FLOAT *dtx, *dty, *dtr;
  FLOAT *dtu, *dtv;

  start = std::chrono::system_clock::now();

  // move over all source particles first
  const int32_t srcsize = npfull*sizeof(FLOAT);
  const int32_t trgsize = npart*sizeof(FLOAT);
  cudaMalloc (&dsx, srcsize);
  cudaMalloc (&dsy, srcsize);
  cudaMalloc (&dss, srcsize);
  cudaMalloc (&dsr, srcsize);
  cudaMalloc (&dtu, srcsize);
  cudaMalloc (&dtv, srcsize);
  cudaMemcpy (dsx, hsx.data(), srcsize, cudaMemcpyHostToDevice);
  cudaMemcpy (dsy, hsy.data(), srcsize, cudaMemcpyHostToDevice);
  cudaMemcpy (dss, hss.data(), srcsize, cudaMemcpyHostToDevice);
  cudaMemcpy (dsr, hsr.data(), srcsize, cudaMemcpyHostToDevice);
  cudaMemset (dtu, 0, trgsize);
  cudaMemset (dtv, 0, trgsize);
  dtx = dsx;
  dty = dsy;
  dtr = dsr;

  for (int32_t nstrm=0; nstrm<nstreams; ++nstrm) {

    // round-robin the GPUs used
    //const int32_t thisgpu = nstrm % ngpus;
    //cudaSetDevice(0);

    const dim3 blocks(npfull/THREADS_PER_BLOCK, 1, 1);
    const dim3 threads(THREADS_PER_BLOCK, 1, 1);

    // move the data

    // launch the kernel
    nvortex_2d_nograds_gpu<<<blocks,threads>>>(nperstrm, dsx,dsy,dss,dsr, 0,dtx,dty,dtr,dtu,dtv);

    // check
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel: %s!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // pull data back down
    cudaMemcpy (htu.data(), dtu, trgsize, cudaMemcpyDeviceToHost);
    cudaMemcpy (htv.data(), dtv, trgsize, cudaMemcpyDeviceToHost);
  }

  // time and report
  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  time = elapsed_seconds.count();
  printf( "  device total time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(4+14*(double)npart)/time);
  printf( "    results ( %g %g %g %g %g %g)\n", htu[0], htv[0], htu[1], htv[1], htu[npart-1], htv[npart-1]);

  // free resources
  cudaFree(dsx);
  cudaFree(dsy);
  cudaFree(dss);
  cudaFree(dsr);
  cudaFree(dtu);
  cudaFree(dtv);

  // compare results
  FLOAT errsum = 0.0;
  FLOAT errmax = 0.0;
  for (int32_t i=0; i<npart; ++i) {
    const FLOAT thiserr = std::pow(htu[i]-htu_cpu[i], 2) + std::pow(htv[i]-htv_cpu[i], 2);
    errsum += thiserr;
    if ((FLOAT)std::sqrt(thiserr) > errmax) {
      errmax = (FLOAT)std::sqrt(thiserr);
      //printf( "    err at %d is %g\n", i, errmax);
    }
  }
  printf( "  total host-device error ( %g ) max error ( %g )\n", std::sqrt(errsum/npart), errmax);
}

