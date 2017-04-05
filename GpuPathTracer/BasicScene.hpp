//
// Created by ashish on 4/4/17.
//

#ifndef GPUPROJECT_BASICSCENE_HPP
#define GPUPROJECT_BASICSCENE_HPP

#include <glad/glad.h>
#include <string>
#include <vector_types.h>

class GLFWwindow;

class BasicScene{
public:
    BasicScene(int width, int height, const std::string &title);
    ~BasicScene();
    void run();

private:
    friend void mousePosCallback(GLFWwindow * window,double posX,double posY);
    friend void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    friend void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    void update(double delta);
    void draw();
    struct kernelInfo{
        dim3 blockSize;
        unsigned int * drawRes;
        int width,height;
    }info;
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


//    float pitch = 0;
//    float roll = 0;
//    float savedCamFov =glm::radians(75.0f);
//    GLuint quadVAO = 0,quadVBO;



};


#endif //GPUPROJECT_BASICSCENE_HPP
