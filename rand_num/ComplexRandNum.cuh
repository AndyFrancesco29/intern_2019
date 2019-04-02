#ifndef _COMPLEXRANDNUM_CUH_
#define _COMPLEXRANDNUM_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <math.h>
#include <CUDACOMPLEX.cuh>

//both have to malloc for the result
void getRandomNumber(CudaComplex *result, int num, int state_num);				//get random number in host, num indicates the number of random number, state_num indicates number of state number
__device__ void getDeviceRand(int id, int N, CudaComplex*result);				//get random number in each thread, id indicates the number of the thread, N indicates number of random numbers in one thread

#endif