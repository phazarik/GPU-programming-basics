#include <stdio.h>

//The GPU function:
__global__ void print_numbers(int N){
	int thread_index = threadIdx.x;
	int block_index = blockIdx.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < N){
		//Include all the GPU tasks inside this 'if' condition for safety.
		printf("thread index (in each block) = %d , block index = %d, thread ID = %d \n", thread_index, block_index, idx);
	}
}

int main(){
	int N = 10;                    //We only want to run it on 10 cores for now
	size_t threads_per_block = 4;  //This is my choice
	size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
	
	print_numbers<<<number_of_blocks, threads_per_block>>>(N);
	//Notice the argument N going inside the brackets.

	cudaDeviceSynchronize();
}
