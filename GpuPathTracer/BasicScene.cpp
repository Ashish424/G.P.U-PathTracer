//
// Created by ashish on 4/4/17.
//

#include "BasicScene.hpp"
#include <GLFW/glfw3.h>
#include "utilfun.hpp"
#include <cassert>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cstring>
#include <glm/gtc/type_ptr.hpp>
//#define GLM_ENABLE_EXPERIMENTAL
//#include <glm/gtx/string_cast.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



float MouseSensitivity = 0.25f;
//quad positions in NDC Space
GLfloat quadVertices[20] = {
        // Positions  // Texture Coords
        -1.0f,  1.0f, 0.99f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.99f, 0.0f, 0.0f,
        1.0f,  1.0f,  0.99f, 1.0f, 1.0f,
        1.0f, -1.0f,  0.99f, 1.0f, 0.0f,
};
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void setPitchAndRoll(CamInfo & cam,float xoffset, float yoffset){
    xoffset *= MouseSensitivity;
    yoffset *= MouseSensitivity;

    cam.yaw   += xoffset;
    cam.pitch += yoffset;



    if (cam.pitch > 89.0f)
        cam.pitch = 89.0f;
    if (cam.pitch < -89.0f)
        cam.pitch = -89.0f;


    //rotate along global y ,local x axis
    cam.front.x = cosf(glm::radians(cam.yaw-90))*cosf(glm::radians(cam.pitch));
    cam.front.y = sinf(glm::radians(cam.pitch));
    cam.front.z = sinf(glm::radians(cam.yaw-90))*cosf(glm::radians(cam.pitch));
    //right vector in x-z plane always(no roll camera)
    cam.right = glm::vec3(-cam.front.z,0,cam.front.x);
    cam.up    = glm::normalize(glm::cross(cam.right,cam.front));
}
BasicScene::BasicScene(int width, int height, const std::string &title):width(width),height(height),updater(*this){

    using std::cout;
    using std::endl;


    //init glfw
    {
        glfwInit();

    }
    mainWindow = uf::createWindow(width,height,title.c_str());
    if(mainWindow == nullptr){
        cout <<"failed to create glfw window,exiting" << endl;
        exit(EXIT_FAILURE);
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
        exit(EXIT_FAILURE);
    }

    //cuda init
    int dId;
    if((dId = uf::findCudaDevice()) < 0){
        exit(EXIT_FAILURE);
    }

    //setup draw texture
    {
        //TODO something unregister maybe
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
    //TODO deletions here too

    size_t numTris = 0;
    //cuda texture for triangles
    {
        using glm::vec4;

        thrust::host_vector<float4> cpuTris1(uf::loadTris("./cube.obj"));
        cout << "num verts: " << cpuTris1.size()<< endl;

//        thrust::host_vector<vec4> cpuTris1(uf::loadTris("filename.obj"));
//        thrust::host_vector<vec4> cpuTris2(uf::loadTris("filename.obj"));
//        thrust::host_vector<float4> cpuTris1(800*600);
//        thrust::host_vector<vec4> cpuTris2(1000);
//        for(size_t i = 0;i<cpuTris1.size();++i){
//            cpuTris1[i] = make_float4(rand()/(float)RAND_MAX,rand()/(float)RAND_MAX,rand()/(float)RAND_MAX,rand()/(float)RAND_MAX);
//        }

//        cpuTris1.insert(cpuTris1.end(), cpuTris2.begin(), cpuTris2.end());

        //TODO see if pinned memory here
        cudaMalloc(&gpuTris,sizeof(float4)*cpuTris1.size());
        cudaMemcpy(gpuTris,thrust::raw_pointer_cast(&cpuTris1[0]),sizeof(float4)*cpuTris1.size(),cudaMemcpyHostToDevice);




        trianglesTex.desc.resType = cudaResourceTypeLinear;
        trianglesTex.desc.res.linear.devPtr = thrust::raw_pointer_cast(&gpuTris[0]);
        trianglesTex.desc.res.linear.desc = cudaCreateChannelDesc<float4>();
        trianglesTex.desc.res.linear.sizeInBytes = sizeof(float4)*cpuTris1.size();


        memset(&trianglesTex.texDesc, 0, sizeof(trianglesTex.texDesc));
        trianglesTex.textureObject = 0;
        trianglesTex.texDesc.filterMode = cudaFilterModePoint;
        trianglesTex.texDesc.normalizedCoords = 0;
        trianglesTex.texDesc.addressMode[0] = cudaAddressModeWrap;

        cudaCreateTextureObject(&trianglesTex.textureObject, &trianglesTex.desc, &trianglesTex.texDesc, NULL);

        numTris = cpuTris1.size();
    }
    //kernel default parameters
    {
        info.dev_drawRes = cudaDestResource;
        info.width = width;
        info.height = height;
        info.blockSize = dim3(16,16,1);
        info.triangleTex = trianglesTex.textureObject;
        info.numTris = numTris;
    }

    //setup camera

    {

        info.cam.dist = height/2;
        info.cam.pitch = 0;
        info.cam.yaw = 0;
        info.cam.aspect = width*1.0f/height;
        info.cam.fov = glm::tan(glm::radians(45.0f));

        //rotate along global y ,local x axis
        info.cam.front.x = cosf(glm::radians(info.cam.yaw-90))*cosf(glm::radians(info.cam.pitch));
        info.cam.front.y = sinf(glm::radians(info.cam.pitch));
        info.cam.front.z = sinf(glm::radians(info.cam.yaw-90))*cosf(glm::radians(info.cam.pitch));
        //right vector in x-z plane always(no roll camera)
        info.cam.right = glm::vec3(-info.cam.front.z,0,info.cam.front.x);
        info.cam.up    = glm::normalize(glm::cross(info.cam.right,info.cam.front));
        info.cam.pos = glm::vec3(0,0,0);

    }




}

BasicScene::~BasicScene() {


    //destroy triangles texture
    {
        checkCudaErrors(cudaDestroyTextureObject(trianglesTex.textureObject));
        checkCudaErrors(cudaFree(gpuTris));

    }

    glfwTerminate();
}

void BasicScene::run() {
    double delta = 0;
    double last = 0;
    glfwSetTime(last);
    while (!glfwWindowShouldClose(mainWindow)) {

        double curr = glfwGetTime();
        delta = curr-last;
        last = curr;

        glfwPollEvents();
        update(delta);

        //TODO see if lower level sync here
        //wait till prev frame done
        checkCudaErrors(cudaStreamSynchronize(0));




        uf::GpuTimer g;
        g.Start();

        launchKernel(info);

        g.Stop();
        std::cout << g.Elapsed() << std::endl;


        cudaArray *texturePtr = nullptr;
        checkCudaErrors(cudaGraphicsMapResources(1, &cudaTexResource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texturePtr, cudaTexResource, 0, 0));

        size_t num_texels =  (size_t)width*height;
        size_t num_values = num_texels * 4;
        size_t size_tex_data = sizeof(GLubyte) * num_values;
        checkCudaErrors(cudaMemcpyToArray(texturePtr, 0, 0, cudaDestResource, size_tex_data, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTexResource, 0));


        draw();

        glfwSwapBuffers(mainWindow);

    }


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

void BasicScene::Updater::operator()(double delta) {

    float camSpeed = 300.0f*(float)delta;

    if(glfwGetKey(prtScn.mainWindow,GLFW_KEY_W)){

        prtScn.info.cam.pos+=prtScn.info.cam.front*camSpeed;
    }
    if(glfwGetKey(prtScn.mainWindow,GLFW_KEY_A)){
        prtScn.info.cam.pos-=prtScn.info.cam.right*camSpeed;
    }
    if(glfwGetKey(prtScn.mainWindow,GLFW_KEY_S)){
        prtScn.info.cam.pos-=prtScn.info.cam.front*camSpeed;
    }
    if(glfwGetKey(prtScn.mainWindow,GLFW_KEY_D)){
        prtScn.info.cam.pos+=prtScn.info.cam.right*camSpeed;
    }

    double xPos,yPos;
    glfwGetCursorPos(prtScn.mainWindow,&xPos,&yPos);
    if(firstMouse){
        firstMouse = false;
        lastX = xPos;
        lastY = yPos;
    }
    float offsetX = float(xPos-lastX);
    float offsetY = float(yPos-lastY);
    lastX = xPos;
    lastY = yPos;


    setPitchAndRoll(prtScn.info.cam,offsetX,offsetY);




}

void BasicScene::update(double delta) {


    updater(delta);


}

void BasicScene::draw() {


    glClearColor(0.5,0.0,0.0,0.0);
    glClear(GL_COLOR_BUFFER_BIT);


    glUseProgram(renderQuad.program);

    //TODO optimize on this binding calls just bind them once since this the only opengl thing you do
    glBindTexture(GL_TEXTURE_2D, renderQuad.tex);
    glActiveTexture(GL_TEXTURE0);

    glUniform1i(renderQuad.texUniform, 0);

    glBindVertexArray(renderQuad.vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

}
