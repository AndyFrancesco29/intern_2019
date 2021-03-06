#include "CUDACOMPLEX.cuh"
#include <math.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"


__host__ __device__ DataType ata(DataType x, DataType y)
{

	if (x == 0.)
	{
		if (y == 0) return(0.);
		else return(1.5707963268);
	}
	else return(atan2(y, x));

}

// square
__host__ __device__ DataType sqr(DataType x)
{

	return(x*x);

}

// default constructor
__host__ __device__ CudaComplex::CudaComplex()
{

	re = im = 0.;

}

// contructor that takes two real numbers
__host__ __device__ CudaComplex::CudaComplex(const DataType x, const DataType y)
{

	re = x;
	im = y;

}

// copy constructor
__host__ __device__ CudaComplex::CudaComplex(const CudaComplex &other)
{

	re = other.re;
	im = other.im;

}

// destructor
__host__ __device__ CudaComplex::~CudaComplex()
{

	// do nothing destructor

}

// equality
__host__ __device__ bool CudaComplex::equal(const DataType x, const DataType y) const
{

	if ((re == x) && (im == y)) return(true);
	return(false);

}

// equal operator
__host__ __device__ bool CudaComplex::operator==(const CudaComplex &other) const
{

	if ((re == other.re) && (im == other.im)) return(true);
	return(false);

}

// difference
__host__ __device__ bool CudaComplex::diff(const DataType x, const DataType y) const
{

	if ((re != x) || (im != y)) return(true);
	return(false);

}

// diff operator
__host__ __device__ bool CudaComplex::operator!=(const CudaComplex &other) const
{

	if ((re != other.re) || (im != other.im)) return(true);
	return(false);

}

// copy
__host__ __device__ void CudaComplex::copy(const DataType x, const DataType y)
{

	re = x;
	im = y;

}

// assignment operator
__host__ __device__ CudaComplex &CudaComplex::operator=(const CudaComplex &other)
{

	if (&other != this)
	{
		re = other.re;
		im = other.im;
	}
	return(*this);

}

__host__ __device__ const CudaComplex &CudaComplex::operator +(void) const
{
	return(*this);
}

__host__ __device__ const CudaComplex CudaComplex::operator -(void) const
{
	return(CudaComplex(-re, -im));
}


// this adds two cmplxs in an object
__host__ __device__ void CudaComplex::add(const CudaComplex &oth1, const CudaComplex &oth2)
{

	re = oth1.re + oth2.re;
	im = oth1.im + oth2.im;

}

// operator to add cmplxs and return the result
__host__ __device__ CudaComplex operator + (const CudaComplex &oth1, const CudaComplex &oth2)
{

	return CudaComplex(oth1.re + oth2.re, oth1.im + oth2.im);

}

// operator to add DataType and cmplx and return the result
__host__ __device__ CudaComplex operator + (const DataType oth1, const CudaComplex &oth2)
{

	return CudaComplex(oth1 + oth2.re, oth2.im);

}
__host__ __device__ CudaComplex operator + (const CudaComplex &oth2, const DataType oth1)
{

	return CudaComplex(oth1 + oth2.re, oth2.im);

}


// append operator
__host__ __device__ CudaComplex &CudaComplex::operator+=(const CudaComplex &other)
{

	re += other.re;
	im += other.im;
	return(*this);

}

// append with DataType operator
__host__ __device__ CudaComplex &CudaComplex::operator+=(const DataType other)
{

	re += other;
	return(*this);
}


// this subtracts two cmplxs in an object
__host__ __device__ void CudaComplex::sub(const CudaComplex &oth1, const CudaComplex &oth2)
{

	re = oth1.re - oth2.re;
	im = oth1.im - oth2.im;

}

// operator to subtract cmplxs and return the result
__host__ __device__ CudaComplex operator - (const CudaComplex &oth1, const CudaComplex &oth2)
{

	return CudaComplex(oth1.re - oth2.re, oth1.im - oth2.im);

}

// operator to subtract DataType and cmplx and return the result
__host__ __device__ CudaComplex operator - (const DataType oth1, const CudaComplex &oth2)
{

	return CudaComplex(oth1 - oth2.re, -oth2.im);

}
__host__ __device__ CudaComplex operator - (const CudaComplex &oth2, const DataType oth1)
{

	return CudaComplex(-oth1 + oth2.re, oth2.im);

}


// dedeappend operator
__host__ __device__ CudaComplex &CudaComplex::operator-=(const CudaComplex &other)
{

	re -= other.re;
	im -= other.im;
	return(*this);

}

// deappend with DataType operator
__host__ __device__ CudaComplex &CudaComplex::operator-=(const DataType other)
{

	re -= other;
	return(*this);

}


// this multiplies two cmplxs in an object
__host__ __device__ void CudaComplex::mul(const CudaComplex &oth1, const CudaComplex &oth2)
{

	re = oth1.re*oth2.re - oth1.im*oth2.im;
	im = oth1.re*oth2.im + oth1.im*oth2.re;

}

// operator to multiply cmplxs and return the result
__host__ __device__ CudaComplex operator * (const CudaComplex &oth1, const CudaComplex &oth2)
{

	return CudaComplex(oth1.re*oth2.re - oth1.im*oth2.im,
		oth1.re*oth2.im + oth1.im*oth2.re);

}

// operator to multiply DataType and cmplx and return the result
__host__ __device__ CudaComplex operator * (const DataType oth1, const CudaComplex &oth2)
{

	return CudaComplex(oth1 * oth2.re, oth1 * oth2.im);

}

__host__ __device__ CudaComplex operator * (const CudaComplex &oth2, const DataType oth1)
{

	return CudaComplex(oth1 * oth2.re, oth1 * oth2.im);

}


// append operator
__host__ __device__ CudaComplex &CudaComplex::operator*=(const CudaComplex &other)
{

	DataType t1;
	t1 = re * other.re - im * other.im;
	im = re * other.im + im * other.re;
	re = t1;
	return(*this);

}


// append with DataType operator
__host__ __device__ CudaComplex &CudaComplex::operator*=(const DataType other)
{

	re *= other;
	im *= other;
	return(*this);

}


// this divs two cmplxs in an object
__host__ __device__ void CudaComplex::div(const CudaComplex &oth1, const CudaComplex &oth2)
{

	re = (oth1.re*oth2.re + oth1.im*oth2.im) / (oth2.re*oth2.re + oth2.im*oth2.im);
	im = (oth2.re*oth1.im - oth1.re*oth2.im) / (oth2.re*oth2.re + oth2.im*oth2.im);

}

// operator to div cmplxs and return the result
__host__ __device__ CudaComplex operator / (const CudaComplex &oth1, const CudaComplex &oth2)
{

	return CudaComplex((oth1.re*oth2.re + oth1.im*oth2.im) / (oth2.re*oth2.re + oth2.im*oth2.im),
		(oth2.re*oth1.im - oth1.re*oth2.im) / (oth2.re*oth2.re + oth2.im*oth2.im));

}

// operator to div DataType and cmplx and return the result
__host__ __device__ CudaComplex operator / (const DataType oth1, const CudaComplex &oth2)
{

	return CudaComplex(oth1*oth2.re / (sqr(oth2.re) + sqr(oth2.im)),
		-oth1 * oth2.im / (sqr(oth2.re) + sqr(oth2.im)));

}
__host__ __device__ CudaComplex operator / (const CudaComplex &oth2, const DataType oth1)
{

	return CudaComplex(oth2.re / oth1, oth2.im / oth1);

}

// div append operator
__host__ __device__ CudaComplex &CudaComplex::operator/=(const CudaComplex &other)
{

	DataType t1;
	t1 = (re*other.re + im * other.im) / (sqr(other.re) + sqr(other.im));
	im = (im*other.re - re * other.im) / (sqr(other.re) + sqr(other.im));
	re = t1;
	return(*this);

}

// dappend with DataType operator
__host__ __device__ CudaComplex &CudaComplex::operator/=(const DataType other)
{

	re /= other;
	im /= other;
	return(*this);

}





// abs
__host__ __device__ DataType abs(const CudaComplex &other)
{

	return(sqrt(other.re*other.re + other.im*other.im));

}

// phase in radians
__host__ __device__ DataType arg(const CudaComplex &other)
{

	return(ata(other.re, other.im));

}

// real part
__host__ __device__ DataType real(const CudaComplex &other)
{

	return(other.re);

}

// imaginary part
__host__ __device__ DataType imag(const CudaComplex &other)
{

	return(other.im);

}

// complex conjugate
__host__ __device__ CudaComplex conj(const CudaComplex &other)
{

	return CudaComplex(other.re, -other.im);

}

// square of the magnitude
// DataType norm(const CudaComplex &other)
//{
//
//    return(other.re*other.re+other.im*other.im);
//
//}
//
//// polar from rect
// CudaComplex polar(const DataType x, const DataType y)
//{
//
//    CudaComplex result(x,y);
//    return CudaComplex(abs(result),arg(result));
//
//}
// CudaComplex polar(const CudaComplex &other)
//{
//
//    return CudaComplex (abs(other),arg(other));
//
//}
//
//// rect from polar
// CudaComplex rect(const DataType mag, const DataType arg)
//{
//
//    return CudaComplex (mag*cos(arg),mag*sin(arg));
//
//}
// CudaComplex rect(const CudaComplex &other)
//{
//
//    return CudaComplex (other.re*cos(other.im),other.re*sin(other.im));
//
//}


// exp
__host__ __device__ CudaComplex exp(const CudaComplex &other)
{

	return CudaComplex(exp(other.re)*cos(other.im), exp(other.re)*sin(other.im));

}