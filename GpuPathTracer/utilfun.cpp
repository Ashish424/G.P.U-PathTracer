//
// Created by ashish on 4/4/17.
//

#include "utilfun.hpp"
#include <cassert>
#include <GLFW/glfw3.h>
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>


using std::cout;
using std::endl;

namespace uf {
    GLFWwindow *createWindow(int width, int height, const char *title) {

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

        return glfwCreateWindow(width, height, title, nullptr, nullptr);

    }

    int initGlad() {
        return gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);

    }

    GLuint createGlTex2DCuda(GLsizei w, GLsizei h, GLint clampX, GLint clampY, GLint minFilter, GLint magFilter) {
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);

// set basic parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

// Create texture data (4-component unsigned byte)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, w, h, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL);

// Unbind the texture
        glBindTexture(GL_TEXTURE_2D, 0);
        return texture;
    }

    int findCudaDevice() {
        int devID = 0;
        int device_count;
        checkCudaErrors(cudaGetDeviceCount(&device_count));

        if (device_count == 0) {
            cout << "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n" << endl;
            return -1;
        }


        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

        if (deviceProp.computeMode == cudaComputeModeProhibited) {
            cout << "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice()" <<endl;
            return -1;
        }

        if (deviceProp.major < 1) {
            cout << "gpuDeviceInit(): GPU device does not support CUDA" << endl;
        }

        checkCudaErrors(cudaSetDevice(devID));
        cout << "CUDA Device ID: "<< devID << " " <<  "name: " << deviceProp.name << endl;
        return devID;

    }
    std::vector<char> fileToCharArr(const std::string & filename,bool printFile) {
//FIXME add error checking for filepath
        std::ifstream inputFileHandle;
        inputFileHandle.open(filename);
        auto start = inputFileHandle.tellg();
        inputFileHandle.seekg(0, std::ios::end);
        auto len = inputFileHandle.tellg() - start;
        inputFileHandle.seekg(start);
        std::vector<char> buffer(len);
        inputFileHandle.read(&buffer[0], len);
        //TODO note len!=gcount!!!!
        assert(len>=inputFileHandle.gcount());
        buffer[inputFileHandle.gcount()] = 0;
        inputFileHandle.close();

        if(printFile) {
            std::cout <<"print len is" << len << std::endl;
            for (int i = 0; i < len; ++i) {
                std::cout << buffer[i];
            }
        }
        return buffer;

    }
    GLuint compileShader(const shDf & sd){

        GLuint Shader = glCreateShader(sd.first);
        glShaderSource(Shader, 1, &sd.second, NULL);
        glCompileShader(Shader);
        GLint success;
        GLchar infoLog[512];
        glGetShaderiv(Shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(Shader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        }
        return Shader;


    }

    GLuint makeProgram(GLuint vS, GLuint fS,bool deleteDetachShaders){
        GLint success = 0;
        GLchar infoLog[512];
        GLuint sp = glCreateProgram();
        glAttachShader(sp, vS);
        glAttachShader(sp, fS);
        glLinkProgram(sp);
        glGetProgramiv(sp, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(sp, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }
        if(deleteDetachShaders) {
            glDetachShader(sp,vS);glDeleteShader(vS);
            glDetachShader(sp,fS);glDeleteShader(fS);
        }
        return sp;
    }
}







