
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <math.h>
#include <ComplexRandNum.cuh>



__global__ void getRand() 
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	int N = 3;
	CudaComplex *temp = (CudaComplex*)malloc(N * sizeof(CudaComplex));
	getDeviceRand(id,N,temp);
	CudaComplex sum=temp[0]+temp[1]+temp[2];
	sum.display();
}


int main()
{
	//cudaFree(devStates);
	//cudaFree(N2);
	////cudaFree(N3);
	//return 0;

	/*int N = 10000;
	CudaComplex *result;
	cudaMallocManaged(&result, N * sizeof(CudaComplex));
	getRandomNumber(result, N, 128);
	result[1000].display();
	return 0;*/

	getRand << <2, 8 >> > ();
	return 0;
}