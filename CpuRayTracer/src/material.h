#ifndef MATERIAL_H
#define MATERIAL_H

#include "../lib/rand48/erand48.h"

#include "ray.hpp"
#include "texture.h"
#include <glm/glm.hpp>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum MaterialType { DIFF, SPEC, EMIT };


class Material {

private:
	MaterialType m_type;
	glm::dvec3 m_colour;
	glm::dvec3 m_emission;
    Texture m_texture;

public:
	Material( MaterialType t=DIFF, glm::dvec3 c=glm::dvec3(1,1,1), glm::dvec3 e=glm::dvec3(0,0,0), Texture tex=Texture() );
	MaterialType get_type() const;
	glm::dvec3 get_colour() const;
	glm::dvec3 get_colour_at(double u, double v) const;
	glm::dvec3 get_emission() const;
	Ray get_reflected_ray( const Ray &r, glm::dvec3 &p, const glm::dvec3 &n, unsigned short *Xi ) const;

};


#endif // MATERIAL_H