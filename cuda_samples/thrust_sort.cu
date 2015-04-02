#include <cuda_runtime.h>
 
#include <thrust/device_vector.h>
#include <thrust/sort.h>
 
 
typedef unsigned int uint;
 
 
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
 
 
__global__ void my_kernel(float* data, unit N, float a, int b, float c)
{
    const uint idx = __umul24( blockIdx.x,blockDim.x ) + threadIdx.x;
    if (idx >= N) { return; }
 
    // irgendetwas tun mit data[index]
}
 
 
 
// Simple code example, sort float array (data) with respect to corresponding keys (uint)
void gpu_C_interface(unsigned int* keys, float* data, int N)
{
    thrust::device_vector<uint> keys_device(keys);
    thrust::device_vector<float> data_device(data);
   
    // do something else with data
    uint blocks, threads;
    computeGridSize(N, 256, &blocks, &threads);
    // asynchronous kernel call
    my_kernel<<<blocks,threads>>>( data_device.data().get(), N, 1,2,3 );
 
    // sort data_device by using the order implied by keys_device
    thrust::sort_by_key( keys_device.begin(), keys_device.begin() + N, data_device.begin(), thrust::greater<uint>() );
   
    // asynchronous memcpy, which is serial with respect to GPU, but asynchronous with respect to CPU
    cudaMemcpyAsync(data, data_device.data().get(), sizeof(float)*N, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(keys, keys_device.data().get(), sizeof(uint)*N, cudaMemcpyDeviceToHost);
   
    cudaDeviceSynchronize(); // wait for memcpy to finish so that CPU code operates on correct data
}

