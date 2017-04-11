//
// Created by ashish on 4/4/17.
//

#ifndef GPUPROJECT_BASICSCENE_HPP
#define GPUPROJECT_BASICSCENE_HPP

#include <glad/glad.h>
#include <string>
#include <vector_types.h>
#include <glm/glm.hpp>

class GLFWwindow;

struct CamInfo{
    glm::vec3 front,right,up,pos;
    float dist;
    float pitch ,yaw;
    float aspect;
    float fov;

};
struct kernelInfo{
    dim3 blockSize;
    unsigned int * dev_drawRes;
    int width,height;
    CamInfo cam;
};
class BasicScene{
public:
    BasicScene(int width, int height, const std::string &title);
    ~BasicScene();
    void run();


    kernelInfo info;



private:
    friend void mousePosCallback(GLFWwindow * window,double posX,double posY);
    friend void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    friend void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    void update(double delta);
    void draw();
    void launchKernel(const kernelInfo & info);


private:
    GLFWwindow * mainWindow;
    struct Quad{
       GLuint tex,vao,vbo,program;GLint texUniform;
    }renderQuad;
    struct cudaGraphicsResource *cudaTexResource;
    //the cuda image to use
    unsigned int * cudaDestResource;
    int width,height;

};


#endif //GPUPROJECT_BASICSCENE_HPP
