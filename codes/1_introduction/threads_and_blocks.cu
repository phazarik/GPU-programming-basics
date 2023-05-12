#include <stdio.h>

//The GPU function:
__global__ void print_numbers(){
	int thread_index = threadIdx.x;
	int block_index = blockIdx.x;
	printf("thread index (in each block) = %d, block index = %d \n", thread_index, block_index);
}

int main(){
    print_numbers<<<2, 2>>>();
    cudaDeviceSynchronize();
}
