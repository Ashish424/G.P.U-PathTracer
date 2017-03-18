//
// Created by ashish on 3/19/17.
//

#include "Camera.hpp"



Camera::Camera(Camera::vec3 position, Camera::vec3 up) : position(position),up(up),front(vec3(0.0f, 0.0f, -1.0f))
{
    right = glm::vec3(-front.z,0,front.x);
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
    front.x = cosf(glm::radians(Yaw-90))*cosf(glm::radians(Pitch));
    front.y = sinf(glm::radians(Pitch));
    front.z = sinf(glm::radians(Yaw-90))*cosf(glm::radians(Pitch));
    //right vector in x-z plane always(no roll camera)
    right = glm::vec3(-front.z,0,front.x);
    up    = glm::normalize(glm::cross(right,front));



    dirty = true;

}

void Camera::setCamVec(Camera::vec3 position, Camera::vec3 up, Camera::vec3 front, Camera::vec3 right) {

    dirty = true;
}
