#include "ComplexRandNum.cuh"
#define PI 3.14159f

__device__ float generate_rand(curandState *state, int i) {
	curandState localstate = state[i];
	float random = curand_uniform(&localstate)*2*PI;
	state[i] = localstate;
	return random;
}

__global__ void setup_kernel(curandState *state, unsigned long int seed) {
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	//printf("%d\n", i);
	curand_init(seed, i, 2, &state[i]);

}

__global__ void kernel(CudaComplex *N, curandState *state, int num_state, int num_rand)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int d = i;
	for (int i; i < num_rand; i += num_state)
	{
		float k1 = generate_rand(state, d);
		float k2 = generate_rand(state, d);
		N[i]=CudaComplex(k1,k2);
	}
}

	void getRandomNumber(CudaComplex *result, int num, int state_num)
	{
		size_t N = num;
		int state_n = state_num;
		curandState* devStates;
		cudaMallocManaged((void**)&devStates, state_n * sizeof(curandState));

		//CudaComplex *N2;
		//cudaMallocManaged(&result, N * sizeof(CudaComplex));
		

		//cudaMemset(N2, 0, sizeof(float)*N);

		int threadsPerBlock = 256;
		//int numBlock = (N + threadsPerBlock - 1) / threadsPerBlock;
		int stateBlock = (state_n + threadsPerBlock - 1) / threadsPerBlock;
		//printf("%d\n", gridsize);

		setup_kernel << <stateBlock, threadsPerBlock >> > (devStates, time(NULL));
		cudaDeviceSynchronize();

		kernel << <stateBlock, threadsPerBlock >> > (result, devStates, state_n, N);
		cudaDeviceSynchronize();

		//cudaMemcpy(N2,N3,sizeof(float)*N,cudaMemcpyDeviceToHost);

		//for (int i = 0; i < N; i++)
		//{
			/*if(N2[i]==N2[100])
				printf("%d\n",i);*/
			//N[i].display();
		//}

		cudaFree(devStates);
		//cudaFree(N2);
		//cudaFree(N3);
	}

__device__ void getDeviceRand(int id,int N,CudaComplex*result)
	{
		curandState localstate;
		float x, y;
		curand_init(1234, id, 0, &localstate);
		for (int i = 0; i < N; i++)
		{
			x = curand_uniform(&localstate) * 2 * PI;
			y = curand_uniform(&localstate) * 2 * PI;
			result[i] = CudaComplex(x,y);
		}
	}