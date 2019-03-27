
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <math.h>


__device__ float generate_rand(curandState *state, int i) {
	curandState localstate = state[i];
	float random = curand_uniform(&localstate);
	state[i] = localstate;
	return random;
}

__global__ void setup_kernel(curandState *state, unsigned long int seed) {
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	//printf("%d\n", i);
	curand_init(seed, i, 2, &state[i]);
	
}

__global__ void kernel(float *N, curandState *state,int num_state, int num_rand)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int d = i;
	for (int i; i < num_rand; i+=num_state)
	{
		float k = generate_rand(state, d);
		N[i] = k;
	}
	//printf("%f\n", k);

}

int main()
{
	int N = 1<<20;
	int state_n = 256;
	curandState* devStates;
	cudaMallocManaged((void**)&devStates, state_n * sizeof(curandState));

	float *N2;
	cudaMallocManaged(&N2, N * sizeof(float));
	//float *N3;
	//cudaMalloc((void**)&N3, sizeof(float)*N);

	cudaMemset(N2, 0, sizeof(float)*N);

	int threadsPerBlock = 256;
	int numBlock = (N + threadsPerBlock - 1) / threadsPerBlock;
	int stateBlock = (state_n + threadsPerBlock - 1) / threadsPerBlock;
	//printf("%d\n", gridsize);

	setup_kernel << <stateBlock, threadsPerBlock >> > (devStates, time(NULL));
	cudaDeviceSynchronize();

	kernel << <stateBlock, threadsPerBlock >> > (N2, devStates,state_n,N);
	cudaDeviceSynchronize();

	//cudaMemcpy(N2,N3,sizeof(float)*N,cudaMemcpyDeviceToHost);

	//for (int i = 0; i < N; i++)
	//{
	//	if(N2[i]==N2[100])
	//		printf("%d\n",i);
	//	printf("%f\n", N2[i]);
	//}

	cudaFree(devStates);
	cudaFree(N2);
	//cudaFree(N3);
	return 0;
}