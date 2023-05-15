#include <stdio.h>
#define N  4
/* Note :It defines a macro named N and assigns it the value 4.
 * When the code is compiled, the preprocessor performs a text-replacement
   of every occurrence of N with the value 4.*/

__global__ void matrixMulGPU( int * a, int * b, int * c ){
  int val = 0;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < N && col < N) {
    for ( int k = 0; k < N; ++k )
      val += a[row * N + k] * b[k * N + col];
    c[row * N + col] = val;
  }
}

//A CPU equivalent of the same thing:
void matrixMulCPU( int * a, int * b, int * c ){
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}

//Function for printing out the matices on screen:
void print_matrix(int *m, const char *name){
  printf("The matrix %s is ..\n", name);
  for( int row = 0; row < N; ++row ){
    for( int col = 0; col < N; ++col ){
      int element = m[row*N + col];
	printf("%d ", element);
    }
    printf("\n");
  }
  printf("\n");
}

//############
//#   MAIN   # 
//############

int main(){


  int *a, *b, *c_cpu, *c_gpu;
  int size = N * N * sizeof (int); // Number of bytes of an N x N matrix

  // Allocate memory
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);

  // Defining the matrices:
  for( int row = 0; row < N; ++row ){
    for( int col = 0; col < N; ++col ){
	//a, b are 1D (flattedned) arrays.
	//that's why, the (i, j) element is actually given by i*N+j
      a[row*N + col] = row;
      b[row*N + col] = col+2;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }
  }

  //Printing out the matrices:
  print_matrix(a, "a");
  print_matrix(b, "b");

  dim3 threads_per_block (16, 16, 1); // A 16 x 16 block threads
  dim3 number_of_blocks ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );
  cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

  // Call the CPU version to check our work
  matrixMulCPU( a, b, c_cpu );

 //Printing out the results:
 printf("Let's look at the results.\n\n");
 print_matrix(c_gpu, "a+b (in GPU)");
 print_matrix(c_cpu, "a+b (in CPU)");

  // Compare the two answers to make sure they are equal
  bool error = false;
  for(int row = 0; row < N && !error; ++row){
    for(int col = 0; col < N && !error; ++col){
      if(c_cpu[row * N + col] != c_gpu[row * N + col]){
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
    }
  }
  if (!error)
    printf("The two results are identical!\n");

  //Free all our allocated memory
  cudaFree(a); cudaFree(b);
  cudaFree( c_cpu ); cudaFree( c_gpu );
}