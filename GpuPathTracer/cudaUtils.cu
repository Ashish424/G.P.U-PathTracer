//
// Created by ashish on 4/6/17.
//

#include "math_functions.h"
//TODO move these headers to cudaUtils.cuh file






using glm::vec3;
using glm::normalize;


enum Refl { DIFF, SPEC, REFR };
struct Ray {
    glm::vec3 origin,dir;
    __device__ Ray(glm::vec3 o, glm::vec3 d) : origin(o), dir(d) {}
};
struct Sphere {

    float rad;				// radius
    glm::vec3 pos, emi, col;	// position, emission, color
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
    __device__ Sphere(float rad,vec3 pos,vec3 emi,vec3 col,Refl refl):rad(rad),pos(pos),emi(emi),col(col),refl(refl){
    }
};



__device__ float clamp(float x, float a, float b);
__device__ int clamp(int x, int a, int b);
__device__ int rgbToInt(float r, float g, float b);
__device__ Ray getCamRayDir(const int px,const int py);







__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}
__device__ int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}



__device__ Ray getCamRayDir(const CamInfo & cam ,const int px,const int py,const int w,const int h){




    //TODO see stepping here
    //objects need to have negative coords relative to camera
    const int xStep = (px - w/2.0f + 0.5)*cam.dist*cam.aspect*cam.fov/w;
    const int yStep = (py - h/2.0f + 0.5)*cam.dist*cam.fov/h;
    glm::vec3 dir = cam.front+cam.right*(1.0f*xStep)+cam.up*(1.0f*yStep);
    //TODO normalize these vectors



    return Ray(cam.pos,normalize(dir));





}