#include "ray.hpp"
#include "camera.hpp"

#include <glm/vec3.hpp>
#include <iostream>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
Camera::Camera(glm::dvec3 position, glm::dvec3 target, int width, int height) {
    using glm::cross;
    using glm::normalize;
    using glm::dvec3;


    m_width = width;
    m_width_recp = 1./m_width;
    m_height = height;
    m_height_recp = 1./m_height;
    m_ratio = (double)m_width/m_height;

    m_position = position;
    m_direction = normalize(target - m_position);
//
    m_x_direction = cross(dvec3(0, 0, 1),-1.0*m_direction);
    m_y_direction = normalize(cross(m_x_direction,(m_direction)));



//    m_x_direction = glm::dvec3(-m_direction.y,m_direction.x,0);
//    m_y_direction = cross(m_direction,m_x_direction);

    using namespace std;
    cout <<"old" << endl;
    cout << "z" << glm::to_string(m_direction) << endl;
    cout << "x"<< glm::to_string(m_x_direction) << endl;
    cout << "y"<< glm::to_string(m_y_direction) << endl;


//    cout <<"new " << endl;
//    cout << glm::to_string(m_x_direction) << endl;
//    cout << glm::to_string(m_y_direction) << endl;




    using std::cout;
    using std::endl;

//    cout << glm::to_string(m_direction)<< endl;
//    cout << glm::to_string(m_x_direction)<< endl;
//    cout << glm::to_string(m_y_direction)<< endl;

    m_x_spacing = (2.0 * m_ratio)/(double)m_width;
    m_y_spacing = (double)2.0/(double)m_height;
    m_x_spacing_half = m_x_spacing * 0.5;
    m_y_spacing_half = m_y_spacing * 0.5;
    dirty = true;

}

int Camera::get_width() { return m_width; }
int Camera::get_height() { return m_height; }

// Returns ray from camera origin through pixel at x,y
Ray Camera::get_ray(int x, int y, bool jitter, unsigned short *Xi) {

    double x_jitter;
    double y_jitter;

    // If jitter == true, jitter point for anti-aliasing
    if (jitter) {
        x_jitter = (erand48(Xi) * m_x_spacing) - m_x_spacing_half;
        y_jitter = (erand48(Xi) * m_y_spacing) - m_y_spacing_half;

    }
    else {
        x_jitter = 0;
        y_jitter = 0;
    }

    glm::dvec3 pixel = m_position + m_direction*2.0;
    pixel = pixel - m_x_direction*m_ratio + m_x_direction*((x * 2 * m_ratio)*m_width_recp) + x_jitter;
    pixel = pixel + m_y_direction - m_y_direction*((y * 2.0)*m_height_recp + y_jitter);
    return Ray(m_position, glm::normalize((pixel-m_position)));
}

void Camera::setPitchAndRoll(GLfloat xoffset, GLfloat yoffset, GLboolean constrainPitch) {



    xoffset *= this->MouseSensitivity;
    yoffset *= this->MouseSensitivity;



    this->Yaw   += xoffset;
    this->Pitch += yoffset;



    // Make sure that when pitch is out of bounds, screen doesn't get flipped
    if(constrainPitch){
        if (this->Pitch > 89.0f)
            this->Pitch = 89.0f;
        if (this->Pitch < -89.0f)
            this->Pitch = -89.0f;
    }


    //rotate along global y ,local x axis
//    front.x = cosf(glm::radians(Yaw-90))*cosf(glm::radians(Pitch));
//    front.y = sinf(glm::radians(Pitch));
//    front.z = sinf(glm::radians(Yaw-90))*cosf(glm::radians(Pitch));
//    right vector in x-z plane always(no roll camera)
//    right = glm::vec3(-front.z,0,front.x);
//    up    = glm::normalize(glm::cross(right,front));



    dirty = true;

}