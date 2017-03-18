#ifndef CAMERA_H
#define CAMERA_H

#include "../lib/rand48/erand48.h"
#include "ray.h"


class Camera {

private:
    int m_width;
    double m_width_recp;
    int m_height;
    double m_height_recp;
    double m_ratio;
    double m_x_spacing;
    double m_x_spacing_half;
    double m_y_spacing;
    double m_y_spacing_half;
    glm::dvec3 m_position;
    glm::dvec3 m_direction;
    glm::dvec3 m_x_direction;
    glm::dvec3 m_y_direction;

public:
    Camera(glm::dvec3 position, glm::dvec3 target, int width, int height);
    int get_width();
    int get_height();
    Ray get_ray(int x, int y, bool jitter, unsigned short *Xi);

};

#endif //CAMERA_H