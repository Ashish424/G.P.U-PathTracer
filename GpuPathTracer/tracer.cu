//
// Created by ashish on 4/4/17.
//

//TODO remove this header
#include "device_launch_parameters.h"


#include "BasicScene.hpp"
#include "cuda_runtime.h"
#include <stdio.h>
#include "math.h"
#include <math_functions.h>
#include <vector_functions.h>
#include <vector_types.h>
#include "vector_functions.h"
#include "device_launch_parameters.h"
#include "cutil_math.h"  // required for float3 vector math
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cudaUtils.cu"


#define EntrypointSentinel 0x76543210

#define STACK_SIZE  64  // Size of the traversal stack in local memory.
texture<float4, 1, cudaReadModeElementType> bvhNodesTexture;
texture<float4, 1, cudaReadModeElementType> triWoopTexture;
texture<float4, 1, cudaReadModeElementType> triNormalsTexture;
texture<int, 1, cudaReadModeElementType> triIndicesTexture;

__device__ int counter = 0;
__global__ void cudaProcess(const kernelInfo info){




    uint tx = threadIdx.x;
    uint ty = threadIdx.y;
    uint bw = blockDim.x;
    uint bh = blockDim.y;
    uint x = blockIdx.x*bw + tx;
    uint y = blockIdx.y*bh + ty;
    size_t pixelPos = y*info.width+x;
    const glm::vec4 * const triTex = info.triangleTex;
    const size_t triTexSize = info.numVerts;
    int w = info.width;
    int h = info.height;
    if(x == 0 && y ==0 ) {

//        printf("rei tex size is %ld \n",info.numVerts);
        //TODO keep this function and disable it
//        printf("received vars\n");
//        printf("%f\n",info.cam.dist);
//        printf("%f\n",info.cam.fov);
//        printf("%f\n",info.cam.aspect);
//        printf("%d\n",info.width);
//        printf("%d\n",info.height);
//        printf("cam width %f\n",info.cam.dist*info.cam.aspect*info.cam.fov);
//        printf("tri tex size %ld\n",info.numVerts);
    }

    if(x>=w || y>=h)
        return;


    //test texture cuda

//    float4 col  = tex1Dfetch<float4>(info.triangleTex,pixelPos);
//    glm::vec4 newVec = f4tov(col);
//    col = vtof4(newVec);


    u_char r = 255,g = 255,b = 255,a = 255;

    Ray camRay = getCamRayDir(info.cam,x,y,w,h);

    {
        float t;
        int triangle_id;
        int geomtype = -1;

        float tmin = 1e20;
        float tmax = -1e20;

        float d = 1e21;
        float k = 1e21;
        float q = 1e21;
        float inf = t = 1e20;





        // if ray hits bounding box of triangle meshes, intersect ray with all triangles
        //TODO insert bounding box here
        intersectAllTriangles(triTex,camRay, t, triangle_id, triTexSize, geomtype,info.cullBackFaces);







        if(t<inf){
            r = 255;
            g = 0;
            b = 0;
            a = 255;
        }



//         t is distance to closest intersection of ray with all primitives in the scene (spheres, boxes and triangles)
//        return t<inf;

    }



    //sphere test
//    {
//
//        float rad= 300/(sqrt(2.0f)-1);
//        Sphere sp(rad/2,vec3(0.0f, 0,-rad-h/2),vec3(0,0,0),vec3(0.9f, 0.9f, 0.9f ), DIFF);
//        float dist = sp.intersect(camRay);
//
//        if(dist > 0 ){
//            r = 0;
//            g = 255;
//            b = 0;
//            a = 255;
//        }
//    }


    uchar4 c4 = make_uchar4(r, g, b, a);
//    uchar4 c4
    info.dev_drawRes[pixelPos] = rgbToUint(c4.x,c4.y,c4.z);

}

#include <iostream>
void BasicScene::launchKernel(const kernelInfo &info) {

//    using namespace std;
//    cout << width <<" " << height << endl;


     dim3 blocks((info.width+info.blockSize.x)/info.blockSize.x,(info.height+info.blockSize.y)/info.blockSize.y,1);
     cudaProcess<<<blocks,info.blockSize>>>(info);
}

__device__ __inline__ void swap2(int& a, int& b){ int temp = a; a = b; b = temp;}
__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }

__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d){ return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d)	{ return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }

__device__ void intersectBVHandTriangles(const float4 rayorig, const float4 raydir,
                                         const float4* gpuNodes, const float4* gpuTriWoops, const float4* gpuDebugTris, const int* gpuTriIndices,
                                         int& hitTriIdx, float& hitdistance, int& debugbingo, vec3& trinormal, int leafcount, int tricount, bool anyHit)
{
    // assign a CUDA thread to every pixel by using the threadIndex
    // global threadId, see richiesams blogspot
    int thread_index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    ///////////////////////////////////////////
    //// FERMI / KEPLER KERNEL
    ///////////////////////////////////////////

    // BVH layout Compact2 for Kepler, Ccompact for Fermi (nodeOffsetSizeDiv is different)
    // void CudaBVH::createCompact(const BVH& bvh, int nodeOffsetSizeDiv)
    // createCompact(bvh,16); for Compact2
    // createCompact(bvh,1); for Compact

    int traversalStack[STACK_SIZE];

    // Live state during traversal, stored in registers.

    int		rayidx;		// not used, can be removed
    float   origx, origy, origz;    // Ray origin.
    float   dirx, diry, dirz;       // Ray direction.
    float   tmin;                   // t-value from which the ray starts. Usually 0.
    float   idirx, idiry, idirz;    // 1 / ray direction
    float   oodx, oody, oodz;       // ray origin / ray direction

    char*   stackPtr;               // Current position in traversal stack.
    int     leafAddr;               // If negative, then first postponed leaf, non-negative if no leaf (innernode).
    int     nodeAddr;
    int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
    float   hitT;                   // t-value of the closest intersection.
    // Kepler kernel only
    //int     leafAddr2;              // Second postponed leaf, non-negative if none.
    //int     nodeAddr = EntrypointSentinel; // Non-negative: current internal node, negative: second postponed leaf.

    int threadId1; // ipv rayidx

    // Initialize (stores local variables in registers)
    {
        // Pick ray index.

        threadId1 = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));


        // Fetch ray.

        // required when tracing ray batches
        // float4 o = rays[rayidx * 2 + 0];
        // float4 d = rays[rayidx * 2 + 1];
        //__shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.

        origx = rayorig.x;
        origy = rayorig.y;
        origz = rayorig.z;
        dirx = raydir.x;
        diry = raydir.y;
        dirz = raydir.z;
        tmin = rayorig.w;

        // ooeps is very small number, used instead of raydir xyz component when that component is near zero
        float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number
        idirx = 1.0f / (fabsf(raydir.x) > ooeps ? raydir.x : copysignf(ooeps, raydir.x)); // inverse ray direction
        idiry = 1.0f / (fabsf(raydir.y) > ooeps ? raydir.y : copysignf(ooeps, raydir.y)); // inverse ray direction
        idirz = 1.0f / (fabsf(raydir.z) > ooeps ? raydir.z : copysignf(ooeps, raydir.z)); // inverse ray direction
        oodx = origx * idirx;  // ray origin / ray direction
        oody = origy * idiry;  // ray origin / ray direction
        oodz = origz * idirz;  // ray origin / ray direction

        // Setup traversal + initialisation

        traversalStack[0] = EntrypointSentinel; // Bottom-most entry. 0x76543210 (1985229328 in decimal)
        stackPtr = (char*)&traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
        leafAddr = 0;   // No postponed leaf.
        nodeAddr = 0;   // Start from the root.
        hitIndex = -1;  // No triangle intersected so far.
        hitT = raydir.w; // tmax
    }

    // Traversal loop.

    while (nodeAddr != EntrypointSentinel)
    {
        // Traverse internal nodes until all SIMD lanes have found a leaf.

        bool searchingLeaf = true; // required for warp efficiency
        while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
        {
            // Fetch AABBs of the two child nodes.

            // nodeAddr is an offset in number of bytes (char) in gpuNodes array

            float4* ptr = (float4*)((char*)gpuNodes + nodeAddr);
            float4 n0xy = ptr[0]; // childnode 0, xy-bounds (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
            float4 n1xy = ptr[1]; // childnode 1, xy-bounds (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
            float4 nz = ptr[2]; // childnode 0 and 1, z-bounds (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
            // ptr[3] contains indices to 2 childnodes in case of innernode, see below
            // (childindex = size of array during building, see CudaBVH.cpp)

            // compute ray intersections with BVH node bounding box

            /// RAY BOX INTERSECTION
            // Intersect the ray against the child nodes.

            float c0lox = n0xy.x * idirx - oodx; // n0xy.x = c0.lo.x, child 0 minbound x
            float c0hix = n0xy.y * idirx - oodx; // n0xy.y = c0.hi.x, child 0 maxbound x
            float c0loy = n0xy.z * idiry - oody; // n0xy.z = c0.lo.y, child 0 minbound y
            float c0hiy = n0xy.w * idiry - oody; // n0xy.w = c0.hi.y, child 0 maxbound y
            float c0loz = nz.x   * idirz - oodz; // nz.x   = c0.lo.z, child 0 minbound z
            float c0hiz = nz.y   * idirz - oodz; // nz.y   = c0.hi.z, child 0 maxbound z
            float c1loz = nz.z   * idirz - oodz; // nz.z   = c1.lo.z, child 1 minbound z
            float c1hiz = nz.w   * idirz - oodz; // nz.w   = c1.hi.z, child 1 maxbound z
            float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin); // Tesla does max4(min, min, min, tmin)
            float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT); // Tesla does min4(max, max, max, tmax)
            float c1lox = n1xy.x * idirx - oodx; // n1xy.x = c1.lo.x, child 1 minbound x
            float c1hix = n1xy.y * idirx - oodx; // n1xy.y = c1.hi.x, child 1 maxbound x
            float c1loy = n1xy.z * idiry - oody; // n1xy.z = c1.lo.y, child 1 minbound y
            float c1hiy = n1xy.w * idiry - oody; // n1xy.w = c1.hi.y, child 1 maxbound y
            float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
            float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

            // ray box intersection boundary tests:

            float ray_tmax = 1e20;
            bool traverseChild0 = (c0min <= c0max); // && (c0min >= tmin) && (c0min <= ray_tmax);
            bool traverseChild1 = (c1min <= c1max); // && (c1min >= tmin) && (c1min <= ray_tmax);

            // Neither child was intersected => pop stack.

            if (!traverseChild0 && !traverseChild1)
            {
                nodeAddr = *(int*)stackPtr; // fetch next node by popping the stack
                stackPtr -= 4; // popping decrements stackPtr by 4 bytes (because stackPtr is a pointer to char)
            }

                // Otherwise, one or both children intersected => fetch child pointers.

            else
            {
                int2 cnodes = *(int2*)&ptr[3];
                // set nodeAddr equal to intersected childnode index (or first childnode when both children are intersected)
                nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

                // Both children were intersected => push the farther one on the stack.

                if (traverseChild0 && traverseChild1) // store closest child in nodeAddr, swap if necessary
                {
                    if (c1min < c0min)
                        swap2(nodeAddr, cnodes.y);
                    stackPtr += 4;  // pushing increments stack by 4 bytes (stackPtr is a pointer to char)
                    *(int*)stackPtr = cnodes.y; // push furthest node on the stack
                }
            }

            // First leaf => postpone and continue traversal.
            // leafnodes have a negative index to distinguish them from inner nodes
            // if nodeAddr less than 0 -> nodeAddr is a leaf
            if (nodeAddr < 0 && leafAddr >= 0)
            {
                searchingLeaf = false; // required for warp efficiency
                leafAddr = nodeAddr;
                nodeAddr = *(int*)stackPtr;  // pops next node from stack
                stackPtr -= 4;  // decrements stackptr by 4 bytes (because stackPtr is a pointer to char)
            }

            // All SIMD lanes have found a leaf => process them.

            // to increase efficiency, check if all the threads in a warp have found a leaf before proceeding to the
            // ray/triangle intersection routine
            // this bit of code requires PTX (CUDA assembly) code to work properly

            // if (!__any(searchingLeaf)) -> "__any" keyword: if none of the threads is searching a leaf, in other words
            // if all threads in the warp found a leafnode, then break from while loop and go to triangle intersection

            //if(!__any(leafAddr >= 0))
            //    break;

            // if (!__any(searchingLeaf))
            //	break;    /// break from while loop and go to code below, processing leaf nodes

            // NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
            // tried everything with CUDA 4.2 but always got several redundant instructions.

            unsigned int mask; // replaces searchingLeaf

            asm("{\n"
                    "   .reg .pred p;               \n"
                    "setp.ge.s32        p, %1, 0;   \n"
                    "vote.ballot.b32    %0,p;       \n"
                    "}"
            : "=r"(mask)
            : "r"(leafAddr));

            if (!mask)
                break;
        }


        ///////////////////////////////////////////
        /// TRIANGLE INTERSECTION
        //////////////////////////////////////

        // Process postponed leaf nodes.

        while (leafAddr < 0)  /// if leafAddr is negative, it points to an actual leafnode (when positive or 0 it's an innernode)
        {
            // Intersect the ray against each triangle using Sven Woop's algorithm.
            // Woop ray triangle intersection: Woop triangles are unit triangles. Each ray
            // must be transformed to "unit triangle space", before testing for intersection

            for (int triAddr = ~leafAddr;; triAddr += 3)  // triAddr is index in triWoop array (and bitwise complement of leafAddr)
            { // no defined upper limit for loop, continues until leaf terminator code 0x80000000 is encountered

                // Read first 16 bytes of the triangle.
                // fetch first precomputed triangle edge
                float4 v00 = tex1Dfetch(triWoopTexture, triAddr);

                // End marker 0x80000000 (negative zero) => all triangles in leaf processed --> terminate
                if (__float_as_int(v00.x) == 0x80000000)
                    break;

                // Compute and check intersection t-value (hit distance along ray).
                float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;   // Origin z
                float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);  // inverse Direction z
                float t = Oz * invDz;

                if (t > tmin && t < hitT)
                {
                    // Compute and check barycentric u.

                    // fetch second precomputed triangle edge
                    float4 v11 = tex1Dfetch(triWoopTexture, triAddr + 1);
                    float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;  // Origin.x
                    float Dx = dirx * v11.x + diry * v11.y + dirz * v11.z;  // Direction.x
                    float u = Ox + t * Dx; /// parametric equation of a ray (intersection point)

                    if (u >= 0.0f && u <= 1.0f)
                    {
                        // Compute and check barycentric v.

                        // fetch third precomputed triangle edge
                        float4 v22 = tex1Dfetch(triWoopTexture, triAddr + 2);
                        float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                        float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                        float v = Oy + t*Dy;

                        if (v >= 0.0f && u + v <= 1.0f)
                        {
                            // We've got a hit!
                            // Record intersection.

                            hitT = t;
                            hitIndex = triAddr; // store triangle index for shading

                            // Closest intersection not required => terminate.
                            if (anyHit)  // only true for shadow rays
                            {
                                nodeAddr = EntrypointSentinel;
                                break;
                            }

                            // compute normal vector by taking the cross product of two edge vectors
                            // because of Woop transformation, only one set of vectors works

                            //trinormal = cross(Vec3f(v22.x, v22.y, v22.z), Vec3f(v11.x, v11.y, v11.z));  // works
                            trinormal = cross(vec3(v11.x, v11.y, v11.z), vec3(v22.x, v22.y, v22.z));
                        }
                    }
                }
            } // end triangle intersection

            // Another leaf was postponed => process it as well.

            leafAddr = nodeAddr;
            if (nodeAddr < 0)    // nodeAddr is an actual leaf when < 0
            {
                nodeAddr = *(int*)stackPtr;  // pop stack
                stackPtr -= 4;               // decrement with 4 bytes to get the next int (stackPtr is char*)
            }
        } // end leaf/triangle intersection loop
    } // end traversal loop (AABB and triangle intersection)

    // Remap intersected triangle index, and store the result.

    if (hitIndex != -1){
        hitIndex = tex1Dfetch(triIndicesTexture, hitIndex);
        // remapping tri indices delayed until this point for performance reasons
        // (slow texture memory lookup in de triIndicesTexture) because multiple triangles per node can potentially be hit
    }

    hitTriIdx = hitIndex;
    hitdistance = hitT;
}