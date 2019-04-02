//only consider operations between DataType and CudaComplex
//https://stackoverflow.com/questions/9860711/cucomplex-h-and-exp
//add a method to generate a random complex numbr between as cos(N)+jsin (N), N in [0, 2pi]


/*
Copyright (C) 1994 Free Software Foundation
	written by Guido Perrone (perrone@polito.it)
This file is part of the GPClass portable library. This library is free
software; you can redistribute it and/or modify it under the terms of
the GNU Library General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your
option) any later version.  This library is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the GNU Library General Public License for more details.
You should have received a copy of the GNU Library General Public
License along with this library; if not, write to the Free Software
Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#if !defined( __CudaComplex_H )
#define __CudaComplex_H


#if !defined(__MATH_H)
#include <math.h>
#endif
#if !defined( __IOSTREAM_H )
#include <iostream>
#endif  // __IOSTREAM_H


//#if !defined(__gERR_H)
//#include "gerr.h"
//#endif  // __gERR_H


typedef float DataType;
DataType ata(DataType x, DataType y);

class CudaComplex
{

public:
	__host__ __device__ CudaComplex();																// default constructor
	__host__ __device__ CudaComplex(const DataType x, const DataType y = 0.);						// constr w/ number
	__host__ __device__ CudaComplex(const CudaComplex &other);										// copy constr
	__host__ __device__ ~CudaComplex();															// destructor

// -------- comparison -----------
	__host__ __device__ bool equal(const DataType x, const DataType y = 0.) const;					// equality
	__host__ __device__ bool operator==(const CudaComplex &other) const;							// equal operator
	__host__ __device__ const CudaComplex& operator +(void) const;
	__host__ __device__ const CudaComplex operator -(void) const;
	__host__ __device__ bool diff(const DataType x, const DataType y = 0.) const;					// difference
	__host__ __device__ bool operator!=(const CudaComplex &other) const;							// diff operator

// -------- assignement -----------
	__host__ __device__ void copy(const DataType x, const DataType y = 0.);						// copies 2 number in a CudaComplex
	__host__ __device__ CudaComplex &operator=(const CudaComplex &other);							// assignment operator

// -------- addition -----------
	__host__ __device__ void add(const CudaComplex &oth1, const CudaComplex &oth2);									// adds 2 CudaComplex
	__host__ __device__ friend CudaComplex operator + (const CudaComplex &oth1, const CudaComplex &oth2);				// sum of CudaComplex   
	__host__ __device__ friend CudaComplex operator + (const DataType oth1, const CudaComplex &oth2);					// DataType + cmplx
	__host__ __device__ friend CudaComplex operator + (const CudaComplex &oth2, const DataType oth1);					// DataType + cmplx

	__host__ __device__ CudaComplex &operator+=(const CudaComplex &other);												// append operator
	__host__ __device__ CudaComplex &operator+=(const DataType other);													// append with DataType

// -------- subtraction -----------
	__host__ __device__ void sub(const CudaComplex &oth1, const CudaComplex &oth2);										// subs 2 cmplx
	__host__ __device__ friend CudaComplex operator - (const CudaComplex &oth1, const CudaComplex &oth2);				// sub of cmplx
	__host__ __device__ friend CudaComplex operator - (const DataType oth1, const CudaComplex &oth2);					// DataType - cmplx
	__host__ __device__ friend CudaComplex operator - (const CudaComplex &oth2, const DataType oth1);					// - DataType + cmplx

	__host__ __device__ CudaComplex &operator-=(const CudaComplex &other);												// deappend operator
	__host__ __device__ CudaComplex &operator-=(const DataType other);													// deappend with DataType

// -------- multiplication --------
	__host__ __device__ void mul(const CudaComplex &oth1, const CudaComplex &oth2);						 // mul 2 cmplx
	__host__ __device__ friend CudaComplex operator *(const CudaComplex &oth1, const CudaComplex &oth2);	 // mult of cmplx
	__host__ __device__ friend CudaComplex operator * (const DataType oth1, const CudaComplex &oth2);		 // DataType * cmplx
	__host__ __device__ friend CudaComplex operator * (const CudaComplex &oth2, const DataType oth1);		 // DataType * cmplx

	__host__ __device__ CudaComplex &operator*=(const CudaComplex &other);									 // append operator
	__host__ __device__ CudaComplex &operator*=(const DataType other);										 // append with DataType

// -------- division -----------
	__host__ __device__ void div(const CudaComplex &oth1, const CudaComplex &oth2);								    	// divs 2 cmplx
	__host__ __device__ friend CudaComplex operator / (const CudaComplex &oth1, const CudaComplex &oth2);				// div of cmplx
	__host__ __device__ friend CudaComplex operator / (const DataType oth1, const CudaComplex &oth2);					// DataType / cmplx
	__host__ __device__ friend CudaComplex operator / (const CudaComplex &oth2, const DataType oth1);					// DataType / cmplx

	__host__ __device__ CudaComplex &operator/=(const CudaComplex &other);												// div append operator
	__host__ __device__ CudaComplex &operator/=(const DataType other);													// div append with DataType

// -------- math functions -----------
	__host__ __device__ friend DataType abs(const CudaComplex &other);							// abs
	__host__ __device__ friend DataType arg(const CudaComplex &other);							// phase radians
	__host__ __device__ friend DataType real(const CudaComplex &other);						// real part
	__host__ __device__ friend DataType imag(const CudaComplex &other);						// imaginary part
	__host__ __device__ friend CudaComplex conj(const CudaComplex &other);						// the CudaComplex conjugate
	__host__ __device__ friend CudaComplex exp(const CudaComplex &other);						// exponential

// -------- mix functions -----------
	__host__ __device__ void display() const { printf("(%f, %f)",re,im); }           // displays the number

private:
	DataType re,                               // real part
		im;                               // imaginary part
};

#endif