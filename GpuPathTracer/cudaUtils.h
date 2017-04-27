//
// Created by ashish on 4/6/17.
//

#include "cuda_runtime.h"
#include "math_functions.h"
#include "CommomStructs.hpp"
#include "BasicScene.hpp"
#include <glm/glm.hpp>
#include <device_launch_parameters.h>




//TODO move these headers to cudaUtils.cuh file


using glm::vec3;
using glm::vec4;



#define EntrypointSentinel 0x76543210
#define STACK_SIZE  64
#define F32_MIN          (1.175494351e-38f)
#define F32_MAX          (3.402823466e+38f)






//struct Box {
//
//    vec3 min;
//    vec3 max;
//    vec3 emi; // emission
//    vec3 col; // colour
//    Mat refl;
//
//    // ray/box intersection
//    // for theoretical background of the algorithm see
//    // http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
//    // optimised code from http://www.gamedev.net/topic/495636-raybox-collision-intersection-point/
//    __device__ float intersect(const Ray &r) const {
//
//        float epsilon = 0.001f; // required to prevent self intersection
//
//        float3 tmin = (min - r.orig) / r.dir;
//        float3 tmax = (max - r.orig) / r.dir;
//
//        float3 real_min = minf3(tmin, tmax);
//        float3 real_max = maxf3(tmin, tmax);
//
//        float minmax = minf1(minf1(real_max.x, real_max.y), real_max.z);
//        float maxmin = maxf1(maxf1(real_min.x, real_min.y), real_min.z);
//
//        if (minmax >= maxmin) { return maxmin > epsilon ? maxmin : 0; }
//        else return 0;
//    }
//
//    // calculate normal for point on axis aligned box
//    __device__ float3 Box::normalAt(float3 &point) {
//
//        float3 normal = make_float3(0.f, 0.f, 0.f);
//        float min_distance = 1e8;
//        float distance;
//        float epsilon = 0.001f;
//
//        if (fabs(min.x - point.x) < epsilon) normal = make_float3(-1, 0, 0);
//        else if (fabs(max.x - point.x) < epsilon) normal = make_float3(1, 0, 0);
//        else if (fabs(min.y - point.y) < epsilon) normal = make_float3(0, -1, 0);
//        else if (fabs(max.y - point.y) < epsilon) normal = make_float3(0, 1, 0);
//        else if (fabs(min.z - point.z) < epsilon) normal = make_float3(0, 0, -1);
//        else normal = make_float3(0, 0, 1);
//
//        return normal;
//    }
//};


__device__ float clamp(float f, float a, float b);
__device__ int rgbToInt(float r, float g, float b);
__device__ uint rgbToUint(float r, float g, float b);
__device__ Ray getCamRayDir(const CamInfo & cam ,const int px,const int py,const int w,const int h);
//__device__ float3 getTriangleNormal(const cudaTextureObject_t & tex,const size_t triangleIndex);
__device__ float RayTriangleIntersection(const Ray &r, const vec3 &v0, const vec3 &edge1, const vec3 &edge2,bool cullBackFaces);
__device__ void intersectAllSpeheres(const vec4 * sphereTex,const Ray & camRay,float& t_scene, int & sphere_id, const size_t numSpheres, int& geomtype);


inline __device__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}



__device__ int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}
__device__ uint rgbToUint(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (uint(b)<<16) | (uint(g)<<8) | uint(r);
}

//inline __device__ vec4 vtof4(const glm::vec4 & v){
//    return make_vec4(v.x,v.y,v.z,v.w);
//}
//inline __device__ glm::vec4 f4tov(const vec4 & f4){
//    return glm::vec4(f4.x,f4.y,f4.z,f4.w);
//}
//
//inline __device__ float3 vtof3(const vec3 & v){
//    return make_float3(v.x,v.y,v.z);
//}
//inline __device__ vec3 f3tov(const float3 & f3){
//    return vec3(f3.x,f3.y,f3.z);
//}

__device__ Ray getCamRayDir(const CamInfo & cam ,const int px,const int py,const int w,const int h){




    //objects need to have negative coords relative to camera
    const float xStep = (px - w/2.0f + 0.5)*cam.dist*cam.aspect*cam.fov/w;
    const float yStep = (py - h/2.0f + 0.5)*cam.dist*cam.fov/h;

    glm::vec3 dir = cam.front*cam.dist+cam.right*(1.0f*xStep)+cam.up*(1.0f*yStep);
//    float3 dir = vtof3(cam.front*cam.dist+cam.right*(1.0f*xStep)+cam.up*(1.0f*yStep));


    //TODO ray begins at the near plane and add random sampling here
    return Ray(cam.pos+dir,normalize(dir));





}

//__device__ float3 getTriangleNormal(const cudaTextureObject_t & tex,const size_t triangleIndex){
//
//vec4 edge1 = tex1Dfetch<vec4>(tex, triangleIndex * 3 + 1);
//vec4 edge2 = tex1Dfetch<vec4>(tex, triangleIndex * 3 + 2);
//
//// cross product of two triangle edges yields a vector orthogonal to triangle plane
//float3 trinormal = cross(make_float3(edge1.x, edge1.y, edge1.z), make_float3(edge2.x, edge2.y, edge2.z));
//trinormal = normalize(trinormal);
//
//return trinormal;
//}
__device__ float RayTriangleIntersection(const Ray &r,
                                         const vec3 &v0,
                                         const vec3 &edge1,
                                         const vec3 &edge2,bool cullBackFaces){

    vec3 tvec = r.origin - v0;
    vec3 pvec = cross(r.dir, edge2);
    float  det = dot(edge1, pvec);
    if(cullBackFaces && det < 0)
        return -1.0f;

    det = __fdividef(1.0f, det);

    float u = dot(tvec, pvec) * det;

    if (u < 0.0f || u > 1.0f)
        return -1.0f;

    vec3 qvec = cross(tvec, edge1);

    float v = dot(r.dir, qvec) * det;

    if (v < 0.0f || (u + v) > 1.0f)
        return -1.0f;

    return dot(edge2, qvec) * det;
}

__device__ void intersectAllTriangles(const vec4 * tex ,const Ray& camRay, float& t_scene, int & triangle_id, const size_t numVerts, int& geomtype,bool cullBackFaces){
    size_t numTris = numVerts/3;

    for (size_t i = 0; i < numTris; i++)
    {
        vec4 v0    = tex[i*3];
        vec4 edge1 = tex[i*3+1];
        vec4 edge2 = tex[i*3+2];

        float t = RayTriangleIntersection(camRay,vec3(v0.x, v0.y, v0.z),
                                          vec3(edge1.x, edge1.y, edge1.z),
                                          vec3(edge2.x, edge2.y, edge2.z),cullBackFaces);

        //TODO 0.001 magic num
        if (t < t_scene && t > 0.001){
            t_scene = t;triangle_id = i;geomtype = GeoType::TRI;
        }

    }
}



__device__ void intersectAllSpeheres(const Sphere * sphereTex,const Ray & camRay,float& t_scene, int & sphere_id, const size_t numSpheres, int& geomtype){


    //TODO sphere magic number
    float hitSphereDist = 1e20;
    for (size_t i = 0; i < numSpheres; i++)
    {
        //TODO 0.001 magic num
            if ((hitSphereDist = sphereTex[i].intersect(camRay)) && hitSphereDist < t_scene && hitSphereDist > 0.01f){
                t_scene = hitSphereDist; sphere_id = (int)i; geomtype = GeoType::SPHERE;
            }

    }


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



__device__ vec3 intersectRayTriangle(const vec3& v0, const vec3& v1, const vec3& v2, const vec4& rayorig, const vec4& raydir){

    const vec3 rayorig3f = vec3(rayorig.x, rayorig.y, rayorig.z);
    const vec3 raydir3f = vec3(raydir.x, raydir.y, raydir.z);

    const float EPSILON = 0.00001f; // works better
    const vec3 miss(F32_MAX, F32_MAX, F32_MAX);

    float raytmin = rayorig.w;
    float raytmax = raydir.w;

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;

    vec3 tvec = rayorig3f - v0;
    vec3 pvec = cross(raydir3f, edge2);
    float det = dot(edge1, pvec);

    float invdet = 1.0f / det;

    float u = dot(tvec, pvec) * invdet;

    vec3 qvec = cross(tvec, edge1);

    float v = dot(raydir3f, qvec) * invdet;

    if (det > EPSILON)
    {
        if (u < 0.0f || u > 1.0f) return miss; // 1.0 want = det * 1/det  
        if (v < 0.0f || (u + v) > 1.0f) return miss;
        // if u and v are within these bounds, continue and go to float t = dot(...	           
    }

    else if (det < -EPSILON)
    {
        if (u > 0.0f || u < 1.0f) return miss;
        if (v > 0.0f || (u + v) < 1.0f) return miss;
        // else continue
    }

    else // if det is not larger (more positive) than EPSILON or not smaller (more negative) than -EPSILON, there is a "miss"
        return miss;

    float t = dot(edge2, qvec) * invdet;

    if (t > raytmin && t < raytmax)
        return vec3(u, v, t);

    // otherwise (t < raytmin or t > raytmax) miss
    return miss;
}

__device__ void intersectBVHandTriangles(const glm::vec4 rayorig, const vec4 raydir,
                                         const glm::vec4 *gpuNodes, const glm::vec4 *gpuDebugTris,
                                         const int *gpuTriIndices,
                                         int &hitTriIdx, float &hitdistance, vec3 &trinormal, bool needClosestHit){

    int traversalStack[STACK_SIZE];

    float   origx, origy, origz;    // Ray origin.
    float   dirx, diry, dirz;       // Ray direction.
    float   tmin;                   // t-value from which the ray starts. Usually 0.
    float   idirx, idiry, idirz;    // 1 / dir
    float   oodx, oody, oodz;       // orig / dir

    char*   stackPtr;
    int		leafAddr;
    int		nodeAddr;
    int     hitIndex;
    float	hitT;
    int threadId1;

    threadId1 = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));

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

    traversalStack[0] = EntrypointSentinel; // Bottom-most entry. 0x76543210 is 1985229328 in decimal
    stackPtr = (char*)&traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
    leafAddr = 0;   // No postponed leaf.
    nodeAddr = 0;   // Start from the root.
    hitIndex = -1;  // No triangle intersected so far.
    hitT = raydir.w;

    while (nodeAddr != EntrypointSentinel) // EntrypointSentinel = 0x76543210
    {
        // Traverse internal nodes until all SIMD lanes have found a leaf.

        bool searchingLeaf = true; // flag required to increase efficiency of threads in warp
        while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
        {
            vec4* ptr = (vec4*)((char*)gpuNodes + nodeAddr);
            vec4 n0xy = ptr[0]; // childnode 0, xy-bounds (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
            vec4 n1xy = ptr[1]; // childnode 1. xy-bounds (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
            vec4 nz = ptr[2]; // childnodes 0 and 1, z-bounds(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)

            // ptr[3] contains indices to 2 childnodes in case of innernode, see below
            // (childindex = size of array during building, see CudaBVH.cpp)

            // compute ray intersections with BVH node bounding box

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

            float ray_tmax = 1e20;
            bool traverseChild0 = (c0min <= c0max) && (c0min >= tmin) && (c0min <= ray_tmax);
            bool traverseChild1 = (c1min <= c1max) && (c1min >= tmin) && (c1min <= ray_tmax);

            if (!traverseChild0 && !traverseChild1)
            {
                nodeAddr = *(int*)stackPtr; // fetch next node by popping stack
                stackPtr -= 4; // popping decrements stack by 4 bytes (because stackPtr is a pointer to char)
            }

                // Otherwise => fetch child pointers.

            else  // one or both children intersected
            {
                int2 cnodes = *(int2*)&ptr[3];
                // set nodeAddr equal to intersected childnode (first childnode when both children are intersected)
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
            if (nodeAddr < 0 && leafAddr >= 0)  // if leafAddr >= 0 -> no leaf found yet (first leaf)
            {
                searchingLeaf = false; // required for warp efficiency
                leafAddr = nodeAddr;

                nodeAddr = *(int*)stackPtr;  // pops next node from stack
                stackPtr -= 4;  // decrement by 4 bytes (stackPtr is a pointer to char)
            }

            // All SIMD lanes have found a leaf => process them.
            // NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
            // tried everything with CUDA 4.2 but always got several redundant instructions.

            // if (!searchingLeaf){ break;  }

            // if (!__any(searchingLeaf)) break; // "__any" keyword: if none of the threads is searching a leaf, in other words
            // if all threads in the warp found a leafnode, then break from while loop and go to triangle intersection

            // if(!__any(leafAddr >= 0))   /// als leafAddr in PTX code >= 0, dan is het geen echt leafNode
            //    break;

            unsigned int mask; // mask replaces searchingLeaf in PTX code

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

        ///////////////////////////////////////
        /// LEAF NODE / TRIANGLE INTERSECTION
        ///////////////////////////////////////

        while (leafAddr < 0)  // if leafAddr is negative, it points to an actual leafnode (when positive or 0 it's an innernode
        {
            // leafAddr is stored as negative number, see cidx[i] = ~triWoopData.getSize(); in CudaBVH.cpp

            for (int triAddr = ~leafAddr;; triAddr += 3)
            {    // no defined upper limit for loop, continues until leaf terminator code 0x80000000 is encountered

                // Read first 16 bytes of the triangle.
                // fetch first triangle vertex
                vec4 v0f = gpuDebugTris[triAddr + 0];

                // End marker 0x80000000 (= negative zero) => all triangles in leaf processed. --> terminate
                if (__float_as_int(v0f.x) == 0x80000000) break;

                vec4 v1f = gpuDebugTris[triAddr + 1];
                vec4 v2f = gpuDebugTris[triAddr + 2];

                const vec3 v0 = vec3(v0f.x, v0f.y, v0f.z);
                const vec3 v1 = vec3(v1f.x, v1f.y, v1f.z);
                const vec3 v2 = vec3(v2f.x, v2f.y, v2f.z);

                // convert vec4 to vec4

                vec4 rayorigvec4 = vec4(rayorig.x, rayorig.y, rayorig.z, rayorig.w);
                vec4 raydirvec4 = vec4(raydir.x, raydir.y, raydir.z, raydir.w);

                vec3 bary = intersectRayTriangle(v0, v1, v2, rayorigvec4, raydirvec4);

                float t = bary.z; // hit distance along ray

                if (t > tmin && t < hitT)   // if there is a miss, t will be larger than hitT (ray.tmax)
                {
                    hitIndex = triAddr;
                    hitT = t;  /// keeps track of closest hitpoint

                    trinormal = cross(v0 - v1, v0 - v2);

                    if (!needClosestHit){  // shadow rays only require "any" hit with scene geometry, not the closest one
                        nodeAddr = EntrypointSentinel;
                        break;
                    }
                }

            } // triangle

            // Another leaf was postponed => process it as well.

            leafAddr = nodeAddr;

            if (nodeAddr < 0)
            {
                nodeAddr = *(int*)stackPtr;  // pop stack
                stackPtr -= 4;               // decrement with 4 bytes to get the next int (stackPtr is char*)
            }
        } // end leaf/triangle intersection loop
    } // end of node traversal loop

    // Remap intersected triangle index, and store the result.

    if (hitIndex != -1){
        // remapping tri indices delayed until this point for performance reasons
        // (slow global memory lookup in de gpuTriIndices array) because multiple triangles per node can potentially be hit

        hitIndex = gpuTriIndices[hitIndex];
    }

    hitTriIdx = hitIndex;
    hitdistance =  hitT;
}