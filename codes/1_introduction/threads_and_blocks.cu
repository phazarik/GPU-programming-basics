#include <stdio.h>

//The GPU function:
__global__ void print_numbers(){
	int thread_id = threadIdx.x;
	int block_id = blockIdx.x;
	printf("Thread ID = %d , Block ID = %d \n", thread_id, block_id);
}

int main(){
    print_numbers<<<2, 2>>>();
    cudaDeviceSynchronize();
}
