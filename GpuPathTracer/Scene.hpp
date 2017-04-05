//
// Created by ashish on 4/4/17.
//

#ifndef GPUPROJECT_SCENE_HPP
#define GPUPROJECT_SCENE_HPP

#include <glad/glad.h>
#include <string>
class GLFWwindow;

class Scene {
public:
    Scene(int width,int height,const std::string & title = "MainWindow");
    virtual void run();
    virtual ~Scene();

protected:
    GLFWwindow * mainWindow = nullptr;
    struct fixedUpdateSettings{
        float timeStep = 1/60.0f;
        int maxSteps = 5;
    }settings;
    struct GLFWHints{
    }glfwHints;
    int fWidth,fHeight;

private:
    virtual void draw(){
//        std::cout << "drawing here"<< std::endl;
    }
    virtual void fixedUpdate(){
//        std::cout << "fixedupdate" <<std::endl;
    }
    virtual void update(double delta){}


};


#endif //GPUPROJECT_SCENE_HPP
