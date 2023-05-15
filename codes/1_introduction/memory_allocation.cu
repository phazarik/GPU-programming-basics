#include <stdio.h>

//The GPU function that doubles elements in an array:
__global__ void doubleElements(int *a, int N){
  
  //Here N is the size of the array.
  //This is the number of parallel processing that we want.
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < N){
    a[idx] *= 2;
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
  int N = 100;
  int *a;
  size_t size = N * sizeof(int);

  //Allocating this size to *a:
  cudaMallocManaged(&a, size);

  //Defining the array of integers:
  for (int i = 0; i < N; ++i){a[i] = i;}

  size_t threads_per_block = 10;
  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
  //This makes sure that there will be N parallel computations.

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize();

  //Verification:
  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");
  //If everything goes right, this should print TRUE.

  cudaFree(a);
}




  