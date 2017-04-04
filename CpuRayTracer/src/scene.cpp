#include "scene.h"
#include "objects.h"

void Scene::add(Object *object) {
    m_objects.push_back( object );
}

ObjectIntersection Scene::intersect(const Ray &ray) {
    ObjectIntersection isct = ObjectIntersection();
    ObjectIntersection temp;
    long size = m_objects.size();

    for (int i=0; i<size; i++){
        temp = m_objects.at((unsigned)i)->get_intersection(ray);

        if (temp.hit) {
            if (isct.u == 0 || temp.u < isct.u) isct = temp;
        }
    }
    return isct;
}

glm::dvec3 Scene::trace_ray(const Ray &ray, int depth, unsigned short*Xi) {

    //TODO use C++11 random distributions
    ObjectIntersection isct = intersect(ray);

    if (!isct.hit) return glm::dvec3();


    if (isct.m.get_type() == EMIT) return isct.m.get_emission();


    glm::dvec3 colour = isct.m.get_colour();


    double p = colour.x>colour.y && colour.x>colour.z ? colour.x : colour.y>colour.z ? colour.y : colour.z;

    // Russian roulette termination.
    // If random number between 0 and 1 is > p, terminate and return hit object's emmission
    double rnd = erand48(Xi);
    if (++depth>5){
        if (rnd<p*0.9) { // Multiply by 0.9 to avoid infinite loop with colours of 1.0
            colour=colour*(0.9/p);
        }
        else {
            return isct.m.get_emission();
        }
    }

    glm::dvec3 x = ray.origin + ray.direction * isct.u;
    Ray reflected = isct.m.get_reflected_ray(ray, x, isct.n, Xi);

    auto v2= trace_ray(reflected,depth,Xi);
    return glm::dvec3(colour.x*v2.x,colour.y*v2.y,colour.z*v2.z);

}
