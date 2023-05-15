#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result){
  if (result != cudaSuccess){
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void GPU_kernel(parameters, int N){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < N; i += stride){

    Do whatever you want with the parameters.

  }
}

int main(){
  //Set the number of kernels you need to use,
  int N = 1000;
  
  size_t size = N * sizeof(float);

  //Declare pointers etc., allocate their sizes and check if something's wrong.
  float *a;
  checkCuda( cudaMallocManaged(&a, size) );


  size_t threadsPerBlock = 256; //This is a choice you can make.
  size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  GPU-kernel<<<numberOfBlocks, threadsPerBlock>>>(parameters, N);

  checkCuda( cudaGetLastError() );
  checkCuda( cudaDeviceSynchronize() );

  //Free the memory allocations.
  checkCuda( cudaFree(a) );

  //Done!
}

