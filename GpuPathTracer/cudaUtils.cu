//
// Created by ashish on 4/6/17.
//

#include "math_functions.h"

//TODO move these headers to cudaUtils.cuh file


using glm::vec3;





enum Refl { DIFF, SPEC, REFR };
struct Ray {
    float3 origin,dir;
    __device__ Ray(float3 o, float3 d) : origin(o), dir(d) {}
};
struct Sphere {

    float rad;				// radius
    float3 pos, emi, col;	// position, emission, color
    Refl refl;			// reflection type (DIFFuse, SPECular, REFRactive)

    __device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit

        // Ray/sphere intersection
        // Quadratic formula required to solve ax^2 + bx + c = 0
        // Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
        // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0

        float3 op = pos - r.origin;  //
        float t, epsilon = 0.01f;
        float b = dot(op, r.dir);
        float disc = b*b - dot(op, op) + rad*rad; // discriminant
        if (disc<0) return 0; else disc = sqrtf(disc);
        return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0);
    }
    __device__ Sphere(float rad,float3 pos,float3 emi,float3 col,Refl refl):rad(rad),pos(pos),emi(emi),col(col),refl(refl){
    }
};



__device__ int rgbToInt(float r, float g, float b);
__device__ uint rgbToUint(float r, float g, float b);
__device__ float4 vtof4(const glm::vec4 & v);
__device__ glm::vec4 f4tov(const float4 & f4);
__device__ Ray getCamRayDir(const CamInfo & cam ,const int px,const int py,const int w,const int h);
__device__ float3 getTriangleNormal(const cudaTextureObject_t & tex,const size_t triangleIndex);
__device__ float RayTriangleIntersection(const Ray &r,const float3 &v0,const float3 &edge1,const float3 &edge2);
__device__ void intersectAllTriangles(const cudaTextureObject_t & tex,const Ray& r, float& t_scene, int& triangle_id, const size_t numTris, int& geomtype);





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
    const int xStep = (px - w/2.0f + 0.5)*cam.dist*cam.aspect*cam.fov/w;
    const int yStep = (py - h/2.0f + 0.5)*cam.dist*cam.fov/h;
    float3 dir = vtof3(cam.front*cam.dist+cam.right*(1.0f*xStep)+cam.up*(1.0f*yStep));


    //TODO ray begins at the near plane and add random sampling here
    return Ray(vtof3(cam.pos)+dir,normalize(dir));





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
                                         const float3 &v0,
                                         const float3 &edge1,
                                         const float3 &edge2){

    float3 tvec = r.origin - v0;
    float3 pvec = cross(r.dir, edge2);
    float  det = dot(edge1, pvec);

    det = __fdividef(1.0f, det);

    float u = dot(tvec, pvec) * det;

    if (u < 0.0f || u > 1.0f)
        return -1.0f;

    float3 qvec = cross(tvec, edge1);

    float v = dot(r.dir, qvec) * det;

    if (v < 0.0f || (u + v) > 1.0f)
        return -1.0f;

    return dot(edge2, qvec) * det;
}

__device__ void intersectAllTriangles(const cudaTextureObject_t & tex,const Ray& r, float& t_scene, size_t & triangle_id, const size_t numTris, int& geomtype){

    for (size_t i = 0; i < numTris; i++)
    {
        // the triangles are packed into the 1D texture using three consecutive float4 structs for each triangle,
        // first float4 contains the first vertex, second float4 contains the first precomputed edge, third float4 contains second precomputed edge like this:
        // (float4(vertex.x,vertex.y,vertex.z, 0), float4 (egde1.x,egde1.y,egde1.z,0),float4 (egde2.x,egde2.y,egde2.z,0))

        // i is triangle index, each triangle represented by 3 float4s in triangle_texture
        float4 v0    = tex1Dfetch<float4>(tex, i * 3);
        float4 edge1 = tex1Dfetch<float4>(tex, i * 3 + 1);
        float4 edge2 = tex1Dfetch<float4>(tex, i * 3 + 2);

//        // intersect ray with reconstructed triangle
        float t = RayTriangleIntersection(r,make_float3(v0.x, v0.y, v0.z),
                                          make_float3(edge1.x, edge1.y, edge1.z),
                                          make_float3(edge2.x, edge2.y, edge2.z));

        // keep track of closest distance and closest triangle
        // if ray/tri intersection finds an intersection point that is closer than closest intersection found so far
        if (t < t_scene && t > 0.001){
            t_scene = t;triangle_id = i;geomtype = 3;
        }
    }
}
