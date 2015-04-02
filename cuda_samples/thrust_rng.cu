#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <math_constants.h>

typedef unsigned int uint;

__device__ inline float fastMin(float a, float b)  { return (a + b - abs(a - b)) * 0.5f; }
__device__ inline float fastMax(float a, float b)  { return (a + b + abs(a - b)) * 0.5f; }
#define DEG2RAD_CUDA(deg) ((deg)*CUDART_PI_F/180.f)
#define RAD2DEG_CUDA(deg) ((deg)*180.f/CUDART_PI_F)

thrust::device_vector<sample> samples_device;

//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(const uint n, const uint blockSize, uint &numBlocks, uint &numThreads)
{
  numThreads = (blockSize < n) ? blockSize : n;
  numBlocks  = iDivUp(n, numThreads);
}


/* initialize the rng to some seed for all particles */
__global__ void initialize_rng(sample *samples, uint N)
{
  const uint idx = __umul24( blockIdx.x,blockDim.x) + threadIdx.x;
  if (idx >= N) { return; }

  curand_init(idx, 0,0, &samples[idx].rnd_seed);
}
 
 
__global__ void my_kernel(sample* samples, unit N, float a, float b, float c)
{
  const uint idx = __umul24( blockIdx.x,blockDim.x) + threadIdx.x;
  if (idx >= N) { return; }

  float coin = curand_uniform(&samples[idx].rnd_seed);
  if (coin > 0.5) {
    // …
  } else {
    // …
  }
  sample[index].value = coin;
}


void global_interface(sample* samples, int N, bool init)
{
  thrust::device_vector<sample> samples_device(samples);

  uint blocks, threads;
  computeGridSize(N, 256, &blocks, &threads);
  if (init) {
    initialize_rng<<<blocks,threads>>>( samples_device.data().get(), N );
  }
  my_kernel<<<blocks,threads>>>( samples_device.data().get(), N, 1,2,3 );  // asynchronous kernel call
  cudaMemcpyAsync(samples, samples_device.data().get(), sizeof(sample)*N, cudaMemcpyDeviceToHost); // async memcpy, which is serial with respect to GPU (memcpy will happen after kernel call finishes)
  cudaDeviceSynchronize(); // wait for memcpy to finish so that CPU code operates on correct data
}
