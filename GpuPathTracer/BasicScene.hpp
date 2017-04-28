//
// Created by ashish on 4/4/17.
//

#ifndef GPUPROJECT_BASICSCENE_HPP
#define GPUPROJECT_BASICSCENE_HPP

#include <glad/glad.h>
#include <string>
#include <vector_types.h>
#include <glm/glm.hpp>
#include <cstring>
#include "utilfun.hpp"
#include "CommomStructs.hpp"
#include "CpuStructs.hpp"
class GLFWwindow;
class CudaBVH;

void setPitchAndRoll(CamInfo & cam,float xoffset, float yoffset);

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
    void draw();
    void launchKernel(const kernelInfo & info);

//opengl stuff
private:
    GLFWwindow * mainWindow;
    struct Quad{
       GLuint tex,vao,vbo,program;GLint texUniform;
    }renderQuad;
    struct cudaGraphicsResource *cudaTexResource;



    void drawWindow(bool visible);
    //the cuda image to use for drawing
    unsigned int * cudaDestResource;
    int width,height;

//cuda triangles texture
//cuda spheres texture
private:
    glm::vec4 * gpuTris = nullptr;
    Sphere * gpuSpheres = nullptr;
private:

    void update(double delta);
    //update is a functor
    struct Updater{
        Updater(BasicScene & prtScn):prtScn(prtScn){}
        void operator () (double delta);
        bool firstMouse = true;
        double lastX,lastY;
        BasicScene & prtScn;
    }updater;
    CudaBVH * gpuBVH = nullptr;
    CamInfo savecam;

};


#endif //GPUPROJECT_BASICSCENE_HPP
