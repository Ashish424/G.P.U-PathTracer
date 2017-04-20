//
// Created by ashish on 4/6/17.
//

#include "math_functions.h"

//TODO move these headers to cudaUtils.cuh file


using glm::vec3;
using glm::vec4;




enum Refl { DIFF, METAL, SPEC, REFR, COAT };
struct Ray {
    vec3 origin,dir;
    __device__ Ray(vec3 o, vec3 d) : origin(o), dir(d) {}
};
struct Sphere {
    float rad;			// radius
    vec3 pos, emi, col;	// position, emission, color
    Refl refl;			// reflection type (DIFFuse, SPECular, REFRactive)
    __device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit

        // Ray/sphere intersection
        // Quadratic formula required to solve ax^2 + bx + c = 0
        // Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
        // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0

        vec3 op = pos - r.origin;  //
        float t, epsilon = 0.01f;
        float b = dot(op, r.dir);
        float disc = b*b - dot(op, op) + rad*rad; // discriminant
        if (disc<0) return 0; else disc = sqrtf(disc);
        return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0);
    }
    __device__ Sphere(float rad,vec3 pos,vec3 emi,vec3 col,Refl refl):rad(rad),pos(pos),emi(emi),col(col),refl(refl){}
};
//struct Box {
//
//    vec3 min;
//    vec3 max;
//    vec3 emi; // emission
//    vec3 col; // colour
//    Refl refl;
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



__device__ int rgbToInt(float r, float g, float b);
__device__ uint rgbToUint(float r, float g, float b);
__device__ float4 vtof4(const glm::vec4 & v);
__device__ glm::vec4 f4tov(const float4 & f4);
__device__ Ray getCamRayDir(const CamInfo & cam ,const int px,const int py,const int w,const int h);
__device__ float3 getTriangleNormal(const cudaTextureObject_t & tex,const size_t triangleIndex);
__device__ float RayTriangleIntersection(const Ray &r,const float3 &v0,const float3 &edge1,const float3 &edge2);
__device__ void intersectAllTriangles(const vec4 * tex,const Ray& r, float& t_scene, int& triangle_id, const size_t numTris, int& geomtype);





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

__device__ float3 getTriangleNormal(const cudaTextureObject_t & tex,const size_t triangleIndex){

float4 edge1 = tex1Dfetch<float4>(tex, triangleIndex * 3 + 1);
float4 edge2 = tex1Dfetch<float4>(tex, triangleIndex * 3 + 2);

// cross product of two triangle edges yields a vector orthogonal to triangle plane
float3 trinormal = cross(make_float3(edge1.x, edge1.y, edge1.z), make_float3(edge2.x, edge2.y, edge2.z));
trinormal = normalize(trinormal);

return trinormal;
}
__device__ float RayTriangleIntersection(const Ray &r,
                                         const vec3 &v0,
                                         const vec3 &edge1,
                                         const vec3 &edge2){

    vec3 tvec = r.origin - v0;
    vec3 pvec = cross(r.dir, edge2);
    float  det = dot(edge1, pvec);
    if(det < 0)
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

__device__ void intersectAllTriangles(const vec4 * tex ,const Ray& r, float& t_scene, int & triangle_id, const size_t numTris, int& geomtype){
    for (size_t i = 0; i < numTris; i++)
    {
        vec4 v0    = tex[i*3];
        vec4 edge1 = tex[i*3+1];
        vec4 edge2 = tex[i*3+2];

        float t = RayTriangleIntersection(r,vec3(v0.x, v0.y, v0.z),
                                          vec3(edge1.x, edge1.y, edge1.z),
                                          vec3(edge2.x, edge2.y, edge2.z));

        if (t < t_scene && t > 0.001){
            t_scene = t;triangle_id = i;geomtype = 3;
        }

    }
}
