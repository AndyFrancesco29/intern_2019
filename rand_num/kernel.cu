
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

__global__ void setup_kernel(curandState *state, unsigned long int seed, int* count) {
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	//printf("%d\n", i);
	curand_init(seed, i, 2, &state[i]);
	count[i] = i;
}

__global__ void kernel(float *N, curandState *state)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	float k = generate_rand(state, i);
	N[i] = k;
	//printf("%f\n", k);

}

int main()
{
	int N = 40000;
	curandState* devStates;
	cudaMalloc((void**)&devStates, N * sizeof(curandState));

	int *count;
	cudaMallocManaged(&count, sizeof(int)*N);
	cudaMemset(count, 0, sizeof(int)*N);

	float *N2;
	cudaMallocManaged(&N2, N * sizeof(float));
	//float *N3;
	//cudaMalloc((void**)&N3, sizeof(float)*N);

	cudaMemset(N2, 0, sizeof(float)*N);

	int threadsPerBlock = 256;
	int numBlock = (N + threadsPerBlock - 1) / threadsPerBlock;
	//printf("%d\n", gridsize);

	setup_kernel << <numBlock, threadsPerBlock >> > (devStates, time(NULL), count);
	cudaDeviceSynchronize();

	kernel << <numBlock, threadsPerBlock >> > (N2, devStates);
	cudaDeviceSynchronize();

	//cudaMemcpy(N2,N3,sizeof(float)*N,cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		//if(N2[i]==0)
		//	printf("%d\n",i);
		printf("%f\n", N2[i]);
		//if (count[i] != i)
		//	printf("%d\n", i);
	}

	cudaFree(N2);
	//cudaFree(N3);
	return 0;
}