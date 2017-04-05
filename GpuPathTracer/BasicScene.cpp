//
// Created by ashish on 4/4/17.
//

#include "BasicScene.hpp"
#include <GLFW/glfw3.h>
#include "utilfun.hpp"
#include <iostream>
#include <cassert>


#include <cuda_gl_interop.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//quad positions in NDC Space
GLfloat quadVertices[20] = {
        // Positions  // Texture Coords
        -1.0f,  1.0f, 0.99f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.99f, 0.0f, 0.0f,
        1.0f,  1.0f,  0.99f, 1.0f, 1.0f,
        1.0f, -1.0f,  0.99f, 1.0f, 0.0f,
};
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
BasicScene::BasicScene(int width, int height, const std::string &title):width(width),height(height){

    using std::cout;
    using std::endl;

    //init glfw
    {
        glfwInit();

    }
    mainWindow = uf::createWindow(width,height,title.c_str());
    if(mainWindow == nullptr){
        cout <<"failed to create glfw window,exiting" << endl;
        exit(-1);
    }
    glfwMakeContextCurrent(mainWindow);
    //additonal glfw setup
    {
        glfwSetWindowUserPointer(mainWindow, this);
//        glfwSetCursorPosCallback(mainWindow, mousePosCallback);
        glfwSetKeyCallback(mainWindow, keyCallback);
//        glfwSetScrollCallback(mainWindow, scrollCallback);
        glfwSetInputMode(mainWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }

    if(!uf::initGlad()){
        cout <<"failed to init glad,exiting" << endl;
        exit(-1);
    }

    //cuda init
    int dId;
    if((dId = uf::findCudaDevice()) < 0){
        exit(-1);
    }

    //setup texture
    {
        //TODO something unreguster maybe
        renderQuad.tex = uf::createGlTex2DCuda(width, height);
        checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTexResource, renderQuad.tex, GL_TEXTURE_2D,
                                                    cudaGraphicsMapFlagsWriteDiscard));
    }
    //buffer setup
    {
        //TODO delete buffers
        glGenVertexArrays(1,&renderQuad.vao);
        glGenBuffers(1,&renderQuad.vbo);

        glBindBuffer(GL_ARRAY_BUFFER, renderQuad.vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);


        glBindVertexArray(renderQuad.vao);
        glBindBuffer(GL_ARRAY_BUFFER, renderQuad.vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid *) 0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid *) (3 * sizeof(GLfloat)));
        glBindVertexArray(0);
    }
    //shader setup
    {
        //TODO delete shader program here
        using namespace uf;
        auto vsS(fileToCharArr("./quad.vert"));
        auto fsS(fileToCharArr("./quad.frag"));
        renderQuad.program = makeProgram(compileShader(shDf(GL_VERTEX_SHADER,&vsS[0])),compileShader(shDf(GL_FRAGMENT_SHADER,&fsS[0])),true);
        renderQuad.texUniform = glGetUniformLocation(renderQuad.program,"tex");
    }
    //cuda buffer setup
    {
        //TODO delete cuda malloc here
        size_t num_texels =  (size_t)width*height;
        size_t num_values = num_texels * 4;
        size_t size_tex_data = sizeof(GLubyte) * num_values;

        //TODO pinned memory
        checkCudaErrors(cudaMalloc((void **)&cudaDestResource, size_tex_data));
    }
    //kernel default parameters
    {
        info.drawRes = cudaDestResource;
        info.width = width;
        info.height = height;
        info.blockSize = dim3(16,16,1);
    }



}

BasicScene::~BasicScene() {

    glfwTerminate();
}

void BasicScene::run() {
    double delta = 0;
    double last = 0;
    glfwSetTime(last);
    while (!glfwWindowShouldClose(mainWindow)) {

        //wait till prev frame done
        checkCudaErrors(cudaThreadSynchronize());


//        pass in the cudaDestResource

        launchKernel(info);

        cudaArray *texturePtr = nullptr;
        checkCudaErrors(cudaGraphicsMapResources(1, &cudaTexResource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texturePtr, cudaTexResource, 0, 0));

        size_t num_texels =  (size_t)width*height;
        size_t num_values = num_texels * 4;
        size_t size_tex_data = sizeof(GLubyte) * num_values;
        checkCudaErrors(cudaMemcpyToArray(texturePtr, 0, 0, cudaDestResource, size_tex_data, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTexResource, 0));




        double curr = glfwGetTime();
        delta = curr-last;
        last = curr;
        glfwPollEvents();

        update(delta);
        draw();

        glfwSwapBuffers(mainWindow);

    }

//
//    // map vertex buffer object for acces by CUDA
//    cudaGLMapBufferObject((void**)&dptr, vbo);
//
//    //clear all pixels:

//
//    // RAY TRACING:
//    launchKernel(info);
//     dim3 grid(WINDOW / block.x, WINDOW / block.y, 1);
//     dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors
//    dim3 block(16, 16, 1);
//    dim3 grid(width / block.x, height / block.y, 1);
//
//    // launch CUDA path tracing kernel, pass in a hashed seed based on number of frames
//    render_kernel << < grid, block >> >(dptr, accumulatebuffer, total_number_of_triangles, frames, WangHash(frames), scene_aabbox_max, scene_aabbox_min);  // launches CUDA render kernel from the host
//
//    cudaThreadSynchronize();
//
//    // unmap buffer
//    cudaGLUnmapBufferObject(vbo);
//    //glFlush();



//    glBindBuffer(GL_ARRAY_BUFFER, vbo);
//    glVertexPointer(2, GL_FLOAT, 12, 0);
//    glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
//
//    glEnableClientState(GL_VERTEX_ARRAY);
//    glEnableClientState(GL_COLOR_ARRAY);
//    glDrawArrays(GL_POINTS, 0, width * height);
//    glDisableClientState(GL_VERTEX_ARRAY);
//
//    glutSwapBuffers();

}



void mousePosCallback(GLFWwindow *window, double posX, double posY) {

}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    auto sn  = (BasicScene * )glfwGetWindowUserPointer(window);
    assert(sn!= nullptr && sn->mainWindow == window);

    if(action == GLFW_PRESS) {
        if (key == GLFW_KEY_R) {

        }
        else if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
    }

}

void scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {

}

void BasicScene::update(double delta) {

}

void BasicScene::draw() {


    glClear(GL_COLOR_BUFFER_BIT);


    glUseProgram(renderQuad.program);

    glBindTexture(GL_TEXTURE_2D, renderQuad.tex);
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(renderQuad.texUniform, 0);

    glBindVertexArray(renderQuad.vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

}
