#include <stdio.h>
#include <assert.h>

//The following function is used to handle errors.
inline cudaError_t checkCuda(cudaError_t result){
  if (result != cudaSuccess){
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

//The following functions is used to initialize vectors.
//It takes a float and puts it in the i-th location of the array 'a' in each iteration.
void initWith(float num, float *a, int N){
  for(int i = 0; i < N; ++i){a[i] = num;}
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < N; i += stride){

    //The addition of individual components happen here.
    result[i] = a[i] + b[i];

  }
}

//For validation:
void checkElementsAre(float target, float *array, int N){
  for(int i = 0; i < N; i++){
    if(array[i] != target){
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main(){
  const int N = 2<<20;
  
  /* Note : 2<<20 shifts the binary representation of 2 20 positions to the left,
   * effectively multiplying it by 2^20, which equals 1,048,576.
   * Therefore, N is assigned the value 1,048,576. */
  
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  checkCuda( cudaMallocManaged(&a, size) );
  checkCuda( cudaMallocManaged(&b, size) );
  checkCuda( cudaMallocManaged(&c, size) );

  initWith(3, a, N); //It's a vector with two components, both are 3.
  initWith(4, b, N); //It's a vector with two components, both are 4.
  initWith(0, c, N); //It's a vector with two components, both are zero, initially.

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  checkCuda( cudaGetLastError() );
  checkCuda( cudaDeviceSynchronize() );

  checkElementsAre(7, c, N); //Checking if all the elements are 7 or not.

  checkCuda( cudaFree(a) );
  checkCuda( cudaFree(b) );
  checkCuda( cudaFree(c) );
}

