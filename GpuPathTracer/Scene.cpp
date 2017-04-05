//
// Created by ashish on 4/4/17.
//

#include <glad/glad.h>
#include "Scene.hpp"
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cfloat>
#include <GLFW/glfw3.h>
#include <iostream>

using namespace std;

Scene::Scene(int width,int height,const std::string & title) :fWidth(width),fHeight(height){

    std::ios_base::sync_with_stdio(false);
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 0);
    glfwWindowHint(GLFW_RED_BITS, 8);
    glfwWindowHint(GLFW_GREEN_BITS, 8);
    glfwWindowHint(GLFW_BLUE_BITS, 8);
    glfwWindowHint(GLFW_ALPHA_BITS, 8);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

#ifdef MY_DEBUG_OPENGL_APP
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT,GLFW_TRUE);
#endif



// Create a GLFWwindow object
    mainWindow = glfwCreateWindow(width,height,title.c_str(), nullptr, nullptr);
    if (mainWindow == nullptr) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(mainWindow);


    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
    {
        cout << "Failed to initialize OpenGL context" << endl;
        exit(-1);
    }




}

void Scene::run() {

    double last = 0,accumulator = 0;
    glfwSetTime(last);
    double delta = 0.0f;
    while (!glfwWindowShouldClose(mainWindow)) {


        double curr = glfwGetTime();
        delta = curr-last;
        last = curr;
        accumulator+=delta;

        glfwPollEvents();


        const int nSteps = static_cast<int>(std::floor(accumulator / settings.timeStep));


        if (nSteps > 0)
        {
            accumulator -= nSteps * settings.timeStep;
        }


        assert(accumulator < settings.timeStep + FLT_EPSILON);

        //avoid spiral of death
        const int nStepsClamped = std::min(nSteps, settings.maxSteps);
        for (int i = 0; i < nStepsClamped; ++i)fixedUpdate();

        update(delta);
        draw();
        glfwSwapBuffers(mainWindow);

    }





}


//void Scene::internalUpdate(double deltaTime) {



//TODO refer these links for implementation
//http://www.unagames.com/blog/daniele/2010/06/fixed-time-step-implementation-box2d
//http://saltares.com/blog/games/fixing-your-timestep-in-libgdx-and-box2d/
//http://plaincode.blogspot.in/2012/05/fixed-timestep-in-cocos2d-x-with-box2d.html



//}

Scene::~Scene() {
    glfwTerminate();
}