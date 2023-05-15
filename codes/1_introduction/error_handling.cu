#include <stdio.h>

__global__ void doubleElements(int *a, int N){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  //Error #1:
  //Here, the code is attempting to access an element outside the range of `a`.
  
  for (int i = idx; i < 10000; i += stride){
    a[i] *= 2;
  }
}

//CPU function that checks whether the elements of *a are doubled or not:
bool checkElementsAreDoubled(int *a, int N){
  for (int i = 0; i < N; ++i){
    if (a[i] != i*2){
      return false;
      printf("%d", i);
    }
  }
  return true;
}

int main(){
  int N = 1000;

  int *a;
  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);
  for (int i = 0; i < N; ++i){a[i] = i;}

  //Error #2:
  //The number of threads per block can have a maximum value of 1024
  //Here I am manually setting it to 2048.

  size_t threads_per_block = 2048;
  //size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  //cudaDeviceSynchronize();

  //We need to define 'cudaError_t' types to catch these variables.
  //This is how you catch errors for both the kernel launch above and any
  //errors that occur during the asynchronous `doubleElements` kernel execution.

  cudaError_t syncErr = cudaGetLastError();
  cudaError_t asyncErr = cudaDeviceSynchronize();
  if (syncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(syncErr));
  if (asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}




  