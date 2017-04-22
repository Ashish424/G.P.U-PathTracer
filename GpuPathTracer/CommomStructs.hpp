//
// Created by ashish on 4/21/17.
//

#ifndef GPUPROJECT_COMMOMSTRUCTS_H_HPP
#define GPUPROJECT_COMMOMSTRUCTS_H_HPP

#include <glm/glm.hpp>
using glm::vec3;
using glm::vec4;
enum GeoType { TRI, SPHERE,BOX,NONE };
enum Mat { DIFF, METAL, SPEC, REFR, COAT };
struct Ray {
    vec3 origin,dir;
    __host__ __device__ Ray(vec3 o, vec3 d) : origin(o), dir(d) {}
};

struct Sphere {
    glm::vec4 posRad;
   	vec3 emi, col;//emission, color
    Mat mat;
    __device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit

        vec3 op = vec3(posRad) - r.origin;
        float t, epsilon = 0.01f;
        float b = dot(op, r.dir);
        float disc = b*b - dot(op, op) + posRad[3]*posRad[3]; // discriminant
        if (disc<0) return 0; else disc = sqrtf(disc);
        return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0);
    }
    __device__ vec3 getNormal(const vec3 & pos)const{
        return glm::normalize(vec3(pos)-vec3(posRad));
    }


    //TODO testing constructor only remove it
    __host__ __device__ Sphere(const glm::vec4 & posRad):posRad(posRad){}
//    __host__ __device__ Sphere(float rad,vec3 pos,vec3 emi,vec3 col,Mat refl):rad(rad),pos(pos),emi(emi),col(col),refl(refl){}
};


struct Box {

    vec3 min;
    vec3 max;
    vec3 emi;
    vec3 col;
    Mat mat;


    __device__ float intersect(const Ray &r) const {
        return 0;

//        float epsilon = 0.001f;
//        vec3 tmin = (min - r.orig) / r.dir;
//        vec3 tmax = (max - r.orig) / r.dir;
//        vec3 real_min = minf3(tmin, tmax);
//        vec3 real_max = maxf3(tmin, tmax);
//        float minmax = minf1(minf1(real_max.x, real_max.y), real_max.z);
//        float maxmin = maxf1(maxf1(real_min.x, real_min.y), real_min.z);
//        if (minmax >= maxmin) { return maxmin > epsilon ? maxmin : 0; }
//        else return 0;
    }

    // calculate normal for point on axis aligned box
    __device__ vec3 normalAt(const vec3 &point) {

        vec3 normal = vec3(0.f, 0.f, 0.f);
//        float min_distance = 1e8;
//        float distance;
//        float epsilon = 0.001f;
//
//        if (fabs(min.x - point.x) < epsilon)      normal = make_vec3(-1, 0, 0);
//        else if (fabs(max.x - point.x) < epsilon) normal = make_vec3(1, 0, 0);
//        else if (fabs(min.y - point.y) < epsilon) normal = make_vec3(0, -1, 0);
//        else if (fabs(max.y - point.y) < epsilon) normal = make_vec3(0, 1, 0);
//        else if (fabs(min.z - point.z) < epsilon) normal = make_vec3(0, 0, -1);
//        else normal = make_vec3(0, 0, 1);
//
        return normal;
    }
};


#endif //GPUPROJECT_COMMOMSTRUCTS_H_HPP
