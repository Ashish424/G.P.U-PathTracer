#ifndef CAMERA_H
#define CAMERA_H

#include "../lib/rand48/erand48.h"
#include "ray.hpp"
#include <glm/glm.hpp>
#include <glad/glad.h>


class Camera {
public:
    Camera(glm::dvec3 position, glm::dvec3 target, int width, int height);

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

    glm::dvec3 m_direction;
    glm::dvec3 m_x_direction;
    glm::dvec3 m_y_direction;
    bool dirty = false;


public:
    glm::dvec3 m_position;
    void setPitchAndRoll(float xoffset, float yoffset, GLboolean constrainPitch = GL_TRUE);
    bool isDirty(){return dirty;}

    GLfloat Pitch = 0,Yaw = 0;
    float MouseSensitivity = 0.25f;

    int get_width();
    int get_height();
    Ray get_ray(int x, int y, bool jitter, unsigned short *Xi);


};

#endif //CAMERA_H