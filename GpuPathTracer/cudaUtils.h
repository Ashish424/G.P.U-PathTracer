//
// Created by ashish on 4/6/17.
//

#include "cuda_runtime.h"
#include "math_functions.h"
#include "CommomStructs.hpp"
#include "BasicScene.hpp"
#include <glm/glm.hpp>




//TODO move these headers to cudaUtils.cuh file


using glm::vec3;
using glm::vec4;









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

inline __device__ float4 vtof4(const glm::vec4 & v){
    return make_float4(v.x,v.y,v.z,v.w);
}
inline __device__ glm::vec4 f4tov(const float4 & f4){
    return glm::vec4(f4.x,f4.y,f4.z,f4.w);
}

inline __device__ float3 vtof3(const vec3 & v){
    return make_float3(v.x,v.y,v.z);
}
inline __device__ vec3 f3tov(const float3 & f3){
    return vec3(f3.x,f3.y,f3.z);
}

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
//float4 edge1 = tex1Dfetch<float4>(tex, triangleIndex * 3 + 1);
//float4 edge2 = tex1Dfetch<float4>(tex, triangleIndex * 3 + 2);
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
