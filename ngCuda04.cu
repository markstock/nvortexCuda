/*
 * ngCuda03.cpp
 *
 * (c)2022 Mark J. Stock <markjstock@gmail.com>
 *
 * v0.3  use atomics to expose more concurrency
 * v0.4  use float4 to effectively unroll loops
 */

#include <vector>
#include <random>
#include <chrono>

#include <cuda_runtime.h>


// compute using float or double
#define FLOAT float
#define RSQRT rsqrtf

// threads per block (hard coded)
#define THREADS_PER_BLOCK 128

// GPU count limit
#define MAX_GPUS 8

// -------------------------
// compute kernel - GPU
__global__ void ngrav_3d_nograds_gpu(
    const int32_t nSrc,
    const FLOAT* const __restrict__ sx,
    const FLOAT* const __restrict__ sy,
    const FLOAT* const __restrict__ sz,
    const FLOAT* const __restrict__ ss,
    const FLOAT* const __restrict__ sr,
    const int32_t tOffset,
    const FLOAT* const __restrict__ tx,
    const FLOAT* const __restrict__ ty,
    const FLOAT* const __restrict__ tz,
    const FLOAT* const __restrict__ tr,
    FLOAT* const __restrict__ tu,
    FLOAT* const __restrict__ tv,
    FLOAT* const __restrict__ tw) {

  // local "thread" id - this is the target particle
  const int32_t i = tOffset + blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;

  // load sources into shared memory (or not)
  __shared__ FLOAT s_sx[THREADS_PER_BLOCK];
  __shared__ FLOAT s_sy[THREADS_PER_BLOCK];
  __shared__ FLOAT s_sz[THREADS_PER_BLOCK];
  __shared__ FLOAT s_ss[THREADS_PER_BLOCK];
  __shared__ FLOAT s_sr[THREADS_PER_BLOCK];

  // velocity accumulators for target point
  float4 locu = make_float4(0.0f,0.0f,0.0f,0.0f);
  float4 locv = make_float4(0.0f,0.0f,0.0f,0.0f);
  float4 locw = make_float4(0.0f,0.0f,0.0f,0.0f);

  const float tr2 = tr[i]*tr[i];

  // which sources do we iterate over?
  const int32_t jcount = nSrc / gridDim.y;
  const int32_t jstart = blockIdx.y * jcount;

  for (int32_t b=0; b<jcount/THREADS_PER_BLOCK; ++b) {

    const int32_t gidx = jstart + b*THREADS_PER_BLOCK + threadIdx.x;
    s_sx[threadIdx.x] = sx[gidx];
    s_sy[threadIdx.x] = sy[gidx];
    s_sz[threadIdx.x] = sz[gidx];
    s_ss[threadIdx.x] = ss[gidx];
    s_sr[threadIdx.x] = sr[gidx];
    __syncthreads();

    // loop over all source points
    for (int32_t j=0; j<THREADS_PER_BLOCK; j+=4) {
      const int32_t jp1 = j+1;
      const int32_t jp2 = j+2;
      const int32_t jp3 = j+3;
      const float4 dx = make_float4(s_sx[j]-tx[i],s_sx[jp1]-tx[i],s_sx[jp2]-tx[i],s_sx[jp3]-tx[i]);
      const float4 dy = make_float4(s_sy[j]-ty[i],s_sy[jp1]-ty[i],s_sy[jp2]-ty[i],s_sy[jp3]-ty[i]);
      const float4 dz = make_float4(s_sz[j]-tz[i],s_sz[jp1]-tz[i],s_sz[jp2]-tz[i],s_sz[jp3]-tz[i]);
      const float4 vr = make_float4(s_sr[j],s_sr[jp1],s_sr[jp2],s_sr[jp3]);
      const float distsqx = dx.x*dx.x + dy.x*dy.x + dz.x*dz.x + vr.x*vr.x + tr2;
      const float distsqy = dx.y*dx.y + dy.y*dy.y + dz.y*dz.y + vr.y*vr.y + tr2;
      const float distsqz = dx.z*dx.z + dy.z*dy.z + dz.z*dz.z + vr.z*vr.z + tr2;
      const float distsqw = dx.w*dx.w + dy.w*dy.w + dz.w*dz.w + vr.w*vr.w + tr2;
      // this extra flop improves time by >10%
      const float4 invR = make_float4(RSQRT(distsqx), RSQRT(distsqy), RSQRT(distsqz), RSQRT(distsqw));
      const float4 invR2 = make_float4(invR.x*invR.x, invR.y*invR.y, invR.z*invR.z, invR.w*invR.w);
      const float factorx = s_ss[j] * invR.x * invR2.x;
      const float factory = s_ss[jp1] * invR.y * invR2.y;
      const float factorz = s_ss[jp2] * invR.z * invR2.z;
      const float factorw = s_ss[jp3] * invR.w * invR2.w;
      //FLOAT factor = s_ss[j] * RSQRT(distsq) / distsq;
      locu.x += dx.x * factorx;
      locu.y += dx.y * factory;
      locu.z += dx.z * factorz;
      locu.w += dx.w * factorw;
      locv.x += dy.x * factorx;
      locv.y += dy.y * factory;
      locv.z += dy.z * factorz;
      locv.w += dy.w * factorw;
      locw.x += dz.x * factorx;
      locw.y += dz.y * factory;
      locw.z += dz.z * factorz;
      locw.w += dz.w * factorw;
    }
    __syncthreads();
  }

  // save into device view
  // use atomics
  atomicAdd(&tu[i], (locu.x+locu.y+locu.z+locu.w) / (4.0f*3.1415926536f));
  atomicAdd(&tv[i], (locv.x+locv.y+locv.z+locv.w) / (4.0f*3.1415926536f));
  atomicAdd(&tw[i], (locw.x+locw.y+locw.z+locw.w) / (4.0f*3.1415926536f));

  return;
}

// -------------------------
// compute kernel - CPU
__host__ void ngrav_3d_nograds_cpu(
    const int32_t nSrc,
    const FLOAT* const __restrict__ sx,
    const FLOAT* const __restrict__ sy,
    const FLOAT* const __restrict__ sz,
    const FLOAT* const __restrict__ ss,
    const FLOAT* const __restrict__ sr,
    const FLOAT tx,
    const FLOAT ty,
    const FLOAT tz,
    const FLOAT tr,
    FLOAT* const __restrict__ tu,
    FLOAT* const __restrict__ tv,
    FLOAT* const __restrict__ tw) {

  // velocity accumulators for target point
  FLOAT locu = 0.0f;
  FLOAT locv = 0.0f;
  FLOAT locw = 0.0f;

  // loop over all source points
  #pragma omp simd reduction(+:locu,locv)
  for (int32_t j=0; j<nSrc; ++j) {
    FLOAT dx = sx[j] - tx;
    FLOAT dy = sy[j] - ty;
    FLOAT dz = sz[j] - tz;
    FLOAT distsq = dx*dx + dy*dy + dz*dz + sr[j]*sr[j] + tr*tr;
    FLOAT invR = rsqrt(distsq);
    FLOAT invR2 = invR*invR;
    FLOAT factor = ss[j] * invR * invR2;
    locu += dx * factor;
    locv += dy * factor;
    locw += dz * factor;
  }

  // save into device view
  // use atomics?!?
  *tu = locu / (4.0f*3.1415926536f);
  *tv = locv / (4.0f*3.1415926536f);
  *tw = locw / (4.0f*3.1415926536f);

  return;
}

// not really alignment, just minimum block sizes
__host__ int32_t buffer(const int32_t _n, const int32_t _align) {
  // 63,64 returns 1; 64,64 returns 1; 65,64 returns 2
  return _align*(1+(_n-1)/_align);
}

// main program

static void usage() {
  fprintf(stderr, "Usage: ngCuda03 [-n=<number>]\n");
  exit(1);
}

int main(int argc, char **argv) {

  // number of particles/points and gpus
  int32_t npart = 400000;
  int32_t force_ngpus = -1;
  bool compare = false;

  for (int i=1; i<argc; i++) {
    if (strncmp(argv[i], "-n=", 3) == 0) {
      int32_t num = atoi(argv[i]+3);
      if (num < 1) usage();
      npart = num;
    } else if (strncmp(argv[i], "-g=", 3) == 0) {
      int32_t num = atof(argv[i]+3);
      if (num < 1 or num > MAX_GPUS) usage();
      force_ngpus = num;
    } else if (strncmp(argv[i], "-c", 2) == 0) {
      compare = true;
    }
  }

  printf( "performing 3D gravitational direct summation on %d points\n", npart);

  // number of GPUs present
  int32_t ngpus = 1;
  cudaGetDeviceCount(&ngpus);
  if (force_ngpus > 0) ngpus = force_ngpus;
  // number of cuda streams to break work into
  int32_t nstreams = std::min(MAX_GPUS, ngpus);
  printf( "  ngpus ( %d )  and nstreams ( %d )\n", ngpus, nstreams);

  // we parallelize targets over GPUs/streams
  const int32_t ntargperstrm = buffer(npart/nstreams, THREADS_PER_BLOCK*nstreams);
  const int32_t ntargpad = ntargperstrm * nstreams;
  printf( "  ntargperstrm ( %d )  and ntargpad ( %d )\n", ntargperstrm, ntargpad);

  // and on each GPU, we parallelize over THREADS_PER_BLOCK targets and nsrcblocks source blocks
  // number of blocks source-wise (break summations over sources into this many chunks)
  const int32_t nsrcblocks = 32;

  // set stream sizes
  const int32_t nsrcpad = buffer(npart, THREADS_PER_BLOCK*nsrcblocks);
  const int32_t nsrcperblock = nsrcpad / nsrcblocks;
  printf( "  nsrcperblock ( %d )  and nsrcpad ( %d )\n", nsrcperblock, nsrcpad);

  // define the host arrays (for now, sources and targets are the same)
  const int32_t npad = std::max(ntargpad,nsrcpad);
  std::vector<FLOAT> hsx(npad), hsy(npad), hsz(npad), hss(npad), hsr(npad), htu(npad), htv(npad), htw(npad);
  const FLOAT thisstrmag = 1.0 / std::sqrt(npart);
  const FLOAT thisrad    = (2./3.) / std::sqrt(npart);
  //std::random_device dev;
  //std::mt19937 rng(dev());
  std::mt19937 rng(1234);
  std::uniform_real_distribution<FLOAT> xrand(0.0,1.0);
  for (int32_t i = 0; i < npart; ++i)    hsx[i] = xrand(rng);
  for (int32_t i = npart; i < npad; ++i) hsx[i] = 0.0;
  for (int32_t i = 0; i < npart; ++i)    hsy[i] = xrand(rng);
  for (int32_t i = npart; i < npad; ++i) hsy[i] = 0.0;
  for (int32_t i = 0; i < npart; ++i)    hsz[i] = xrand(rng);
  for (int32_t i = npart; i < npad; ++i) hsz[i] = 0.0;
  for (int32_t i = 0; i < npart; ++i)    hss[i] = thisstrmag * (2.0*xrand(rng)-1.0);
  for (int32_t i = npart; i < npad; ++i) hss[i] = 0.0;
  for (int32_t i = 0; i < npart; ++i)    hsr[i] = thisrad;
  for (int32_t i = npart; i < npad; ++i) hsr[i] = thisrad;
  for (int32_t i = 0; i < npad; ++i)     htu[i] = 0.0;
  for (int32_t i = 0; i < npad; ++i)     htv[i] = 0.0;
  for (int32_t i = 0; i < npad; ++i)     htw[i] = 0.0;

  // -------------------------
  // do a CPU version

  if (compare) {
  auto start = std::chrono::system_clock::now();

  #pragma omp parallel for
  for (int32_t i=0; i<npart; ++i) {
    ngrav_3d_nograds_cpu(npart, hsx.data(),hsy.data(),hsz.data(),hss.data(),hsr.data(), hsx[i],hsy[i],hsz[i],hsr[i], &htu[i],&htv[i],&htw[i]);
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  double time = elapsed_seconds.count();

  printf( "  host total time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(6+21*(double)npart)/time);
  for (int i=0; i<4; ++i) {
    printf( "    part %d acc %g %g %g)\n", i, htu[i], htv[i], htw[i]);
  }
  }

  // copy the results into temp vectors
  std::vector<FLOAT> htu_cpu(htu);
  std::vector<FLOAT> htv_cpu(htv);
  std::vector<FLOAT> htw_cpu(htw);

  // -------------------------
  // do the GPU version

  // set device pointers, too
  FLOAT *dsx[MAX_GPUS], *dsy[MAX_GPUS], *dsz[MAX_GPUS], *dss[MAX_GPUS], *dsr[MAX_GPUS];
  FLOAT *dtx[MAX_GPUS], *dty[MAX_GPUS], *dtz[MAX_GPUS], *dtr[MAX_GPUS];
  FLOAT *dtu[MAX_GPUS], *dtv[MAX_GPUS], *dtw[MAX_GPUS];
  cudaStream_t stream[MAX_GPUS];

  // allocate space for all sources, part of targets
  const int32_t srcsize = nsrcpad*sizeof(FLOAT);
  const int32_t trgsize = ntargperstrm*sizeof(FLOAT);
  for (int32_t i=0; i<nstreams; ++i) {
    cudaSetDevice(i);
    cudaStreamCreate(&stream[i]);

    cudaMalloc (&dsx[i], srcsize);
    cudaMalloc (&dsy[i], srcsize);
    cudaMalloc (&dsz[i], srcsize);
    cudaMalloc (&dss[i], srcsize);
    cudaMalloc (&dsr[i], srcsize);
    cudaMalloc (&dtu[i], trgsize);
    cudaMalloc (&dtv[i], trgsize);
    cudaMalloc (&dtw[i], trgsize);
  }

  // to be fair, we start timer after allocation but before transfer
  auto start = std::chrono::system_clock::now();

  // now perform the data movement and setting
  for (int32_t i=0; i<nstreams; ++i) {

    cudaSetDevice(i);

    // move the data
    cudaMemcpyAsync (dsx[i], hsx.data(), srcsize, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync (dsy[i], hsy.data(), srcsize, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync (dsz[i], hsz.data(), srcsize, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync (dss[i], hss.data(), srcsize, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync (dsr[i], hsr.data(), srcsize, cudaMemcpyHostToDevice, stream[i]);
    cudaMemsetAsync (dtu[i], 0, trgsize, stream[i]);
    cudaMemsetAsync (dtv[i], 0, trgsize, stream[i]);
    cudaMemsetAsync (dtw[i], 0, trgsize, stream[i]);
    // now we need to be careful to point to the part of the source arrays that hold
    //   just this GPUs set of target particles
    dtx[i] = dsx[i] + i*ntargperstrm;
    dty[i] = dsy[i] + i*ntargperstrm;
    dtz[i] = dsz[i] + i*ntargperstrm;
    dtr[i] = dsr[i] + i*ntargperstrm;

    // check
    auto memerr = cudaGetLastError();
    if (memerr != cudaSuccess) {
      fprintf(stderr, "Failed to upload data (other): %s!\n", cudaGetErrorString(memerr));
      exit(EXIT_FAILURE);
    }
  }

    const dim3 blocksz(THREADS_PER_BLOCK, 1, 1);
    const dim3 gridsz(ntargperstrm/THREADS_PER_BLOCK, nsrcblocks, 1);

  for (int32_t i=0; i<nstreams; ++i) {
    // launch the kernel
    cudaSetDevice(i);
    ngrav_3d_nograds_gpu<<<gridsz,blocksz,0,stream[i]>>>(nsrcpad, dsx[i],dsy[i],dsz[i],dss[i],dsr[i],
                                               0,dtx[i],dty[i],dtz[i],dtr[i],dtu[i],dtv[i],dtw[i]);

    // check
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel (%d): %s!\n", i, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  for (int32_t i=0; i<nstreams; ++i) {
    // pull data back down
    cudaMemcpyAsync (htu.data() + i*ntargperstrm, dtu[i], trgsize, cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync (htv.data() + i*ntargperstrm, dtv[i], trgsize, cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync (htw.data() + i*ntargperstrm, dtw[i], trgsize, cudaMemcpyDeviceToHost, stream[i]);
  }

  // join streams
  for (int32_t i=0; i<nstreams; ++i) {
    cudaStreamSynchronize(stream[i]);
  }
  //cudaDeviceSynchronize();

  // time and report
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  double time = elapsed_seconds.count();
  printf( "  device total time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(6+22*(double)npart)/time);
  for (int i=0; i<4; ++i) {
    printf( "    part %d acc %g %g %g)\n", i, htu[i], htv[i], htw[i]);
  }

  // free resources, after timer
  for (int32_t i=0; i<nstreams; ++i) {
    cudaFree(dsx[i]);
    cudaFree(dsy[i]);
    cudaFree(dsz[i]);
    cudaFree(dss[i]);
    cudaFree(dsr[i]);
    cudaFree(dtu[i]);
    cudaFree(dtv[i]);
    cudaFree(dtw[i]);
    cudaStreamDestroy(stream[i]);
  }

  // compare results
  if (compare) {
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
}

