//
// Created by ashish on 3/19/17.
//

#ifndef GPUPROJECT_CAMERA_HPP
#define GPUPROJECT_CAMERA_HPP
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <cmath>



class Camera
{
    using vec3 = glm::vec3;
public:


    Camera(vec3 position = vec3(0.0f, 0.0f, 0.0f), vec3 up = vec3(0.0f, 1.0f, 0.0f));
    void setPitchAndRoll(float xoffset, float yoffset, GLboolean constrainPitch = GL_TRUE);
    void setCamVec(vec3 position,vec3 up,vec3 front,vec3 right);
    bool isDirty(){return dirty;}


    vec3 position;
    vec3 up;
    vec3 front;
    vec3 right;
    //fov in radians
    float fov = glm::radians(75.0f);
private:
    GLfloat Pitch = 0,Yaw = 0;
    float MouseSensitivity = 0.25f;
    bool dirty = false;
};


#endif //GPUPROJECT_CAMERA_HPP
