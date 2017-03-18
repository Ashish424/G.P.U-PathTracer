#ifndef OBJECTS_H
#define OBJECTS_H


#include "ray.h"
#include "triangle.h"
#include "kdtree.h"
#include "material.h"
#include "../lib/tiny_obj_loader/tiny_obj_loader.h"
//#include "../lib/fastbvh/BVH.h"

struct ObjectIntersection {
	bool hit;	// If there was an intersection
	double u;	// Distance to intersection along ray
	glm::dvec3 n;		// Normal of intersected face
	Material m;	// Material of intersected face

	ObjectIntersection(bool hit_=false, double u_=0, glm::dvec3 n_=glm::dvec3(), Material m_=Material());
};


class Object {

public:
	glm::dvec3 m_p; // Position
	virtual ObjectIntersection get_intersection(const Ray &r) = 0;
};


class Sphere : public Object {

private:
	double m_r;	// Radius
	Material m_m;	// Material

public:
	Sphere(glm::dvec3 p_, double r_, Material m_);	
	virtual double get_radius();
	virtual Material get_material();

	virtual ObjectIntersection get_intersection(const Ray &r);
};


class Mesh : public Object {

private:
	std::vector<tinyobj::shape_t> m_shapes;
    std::vector<tinyobj::material_t> m_materials;
    std::vector<Material> materials;
    std::vector<Triangle*> tris;
    Material m_m;	// Material
    KDNode *node;
	//BVH bvh;

public:
	Mesh(glm::dvec3 p_, const char* file_path, Material m_);
	virtual ObjectIntersection get_intersection(const Ray &r);

};


#endif // OBJECTS_H