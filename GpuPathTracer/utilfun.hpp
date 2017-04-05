//
// Created by ashish on 4/4/17.
//

#ifndef GPUPROJECT_UTILS_HPP
#define GPUPROJECT_UTILS_HPP


#include <glad/glad.h>
#include <vector>
#include <string>
class GLFWwindow;

namespace uf {
    template<typename T>
    void check(T result, char const *const func, const char *const file, int const line) {
        if (result) {
          //FIXME enable this
//        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
//                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
//        DEVICE_RESET
//         Make sure we call CUDA Device Reset before exiting
//                exit(EXIT_FAILURE);
        }
    }

#define checkCudaErrors(val)           uf::check ( (val), #val, __FILE__, __LINE__ )

    GLFWwindow *createWindow(int width, int height, const char *title);

    int initGlad();

    GLuint createGlTex2DCuda(GLsizei w, GLsizei h, GLint clampX = GL_CLAMP_TO_EDGE, GLint clampY = GL_CLAMP_TO_EDGE,
                         GLint minFilter = GL_NEAREST, GLint magFilter = GL_NEAREST);

    int findCudaDevice();

    std::vector<char> fileToCharArr(const std::string & filename,bool printFile = false);
    typedef std::pair<GLenum,const char *> shDf;
    GLuint compileShader(const shDf & sd);
    GLuint compileShader(const shDf & sd);
    GLuint makeProgram(GLuint vS,GLuint fS,bool deleteDetachShaders);

}
#endif //GPUPROJECT_UTILS_HPP
