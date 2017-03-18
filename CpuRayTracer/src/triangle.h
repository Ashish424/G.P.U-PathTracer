#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <algorithm>
#include <cfloat>


#include "aabbox.h"
#include "material.h"
#include <glm/glm.hpp>




struct Triangle {
    glm::dvec3 v0, v1, v2;     // Vertex world space coordinates
    glm::dvec3 e1, e2;         // Edge 1 and edge 2 of triangle
    glm::dvec3 n, t0, t1, t2;  // Triangle normal and texture coordinates
    Material *m;        // Pointer to material

    Triangle(glm::dvec3 v0_, glm::dvec3 v1_, glm::dvec3 v2_, glm::dvec3 t0_=glm::dvec3(), glm::dvec3 t1_=glm::dvec3(), glm::dvec3 t2_=glm::dvec3(), Material *m_=NULL){
        v0=v0_, v1=v1_, v2=v2_, e1=v1-v0, e2=v2-v0, n=normalize(cross(e1,e2));
        t0=t0_, t1=t1_, t2=t2_;
        m=m_;
    }

    // Returns axis aligned bounding box that contains the triangle
    AABBox get_bounding_box(){
        glm::dvec3 bl = glm::dvec3(
                std::min (std::min(v0.x, v1.x), v2.x ) ,
                std::min (std::min(v0.y, v1.y), v2.y ) ,
                std::min (std::min(v0.z, v1.z), v2.z )
        );
        glm::dvec3 tr = glm::dvec3(
                std::max (std::max(v0.x, v1.x), v2.x ) ,
                std::max (std::max(v0.y, v1.y), v2.y ) ,
                std::max (std::max(v0.z, v1.z), v2.z )
        );

        return AABBox(bl, tr);
    }

    // Returns the midpoint of the triangle
    glm::dvec3 get_midpoint(){
        return (v0 + v1 + v2)/3.0;
    }

    // Checks if ray intersects with triangle. Returns true/false.
    bool intersect(Ray ray, double &t, double tmin, glm::dvec3 &norm) const {

        double u, v, t_temp=0;

        glm::dvec3 pvec = cross(ray.direction,e2);
        double det = dot(e1,pvec);
        if (det == 0) return false;
        double invDet = 1. / det;
        glm::dvec3 tvec = ray.origin - v0;
        u = dot(tvec,pvec) * invDet;
        if (u < 0 || u > 1) return false;
        glm::dvec3 qvec = cross(tvec,e1);
        v = dot(ray.direction,qvec) * invDet;
        if (v < 0 || u + v > 1) return false;
        t_temp = dot(e2,qvec) * invDet; // Set distance along ray to intersection
        if (t_temp < tmin) {
            if (t_temp > 1e-9 ){    // Fairly arbritarily small value, scared to change
                t = t_temp;         // it as it works.
                norm = n;
                return true;
            }
        }
        return false;
    }

    // Returns barycentric coordinates of point p on the triangle
    glm::dvec3 barycentric(glm::dvec3 p){
        glm::dvec3 v2_ = p - v0;
        double d00 = dot(e1,e1);
        double d01 = dot(e1,e2);
        double d11 = dot(e2,e2);
        double d20 = dot(v2_,e1);
        double d21 = dot(v2_,e2);
        double d = d00*d11 - d01*d01;
        double v = (d11*d20 - d01*d21) / d;
        double w = (d00*d21 - d01*d20) / d;
        double u = 1 - v - w;
        return glm::dvec3(u, v, w);
    }

    // Returns the colour at point p on the triangle
    glm::dvec3 get_colour_at(glm::dvec3 p){
        if(m==NULL) return glm::dvec3(1,0,1);

        glm::dvec3 b = barycentric(p);
        glm::dvec3 c = glm::dvec3();
        c = c + (t0 * b.x);
        c = c + (t1 * b.y);
        c = c + (t2 * b.z);

        return m->get_colour_at(c.x, c.y);
    }

};

#endif // TRIANGLE_H