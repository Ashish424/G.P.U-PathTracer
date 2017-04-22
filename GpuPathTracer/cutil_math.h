   /*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

/*
    This file implements common mathematical operations on vector types
    (float3, float4 etc.) since these are not provided as standard by CUDA.

    The syntax is modelled on the Cg standard library.
*/

#ifndef CUTIL_MATH_H
#define CUTIL_MATH_H

#include "cuda_runtime.h"

////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef __CUDACC__
#include <math.h>

inline float fminf(float a, float b)
{
  return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
  return a > b ? a : b;
}

inline int max(int a, int b)
{
  return a > b ? a : b;
}

inline int min(int a, int b)
{
  return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif

// float functions
////////////////////////////////////////////////////////////////////////////////

// lerp
inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// clamp
inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}



#endif
