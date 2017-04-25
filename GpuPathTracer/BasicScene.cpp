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
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#include "CommomStructs.hpp"
#include "BVH.hpp"


float MouseSensitivity = 0.25f;
float moveSpeed = 10.0f;
//value setted on basis of distance from the camera
float scrollSensitivity = 0.1f;
//quad positions in NDC Space
GLfloat quadVertices[20] = {
        // Positions  // Texture Coords
        -1.0f,  1.0f, 0.99f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.99f, 0.0f, 0.0f,
        1.0f,  1.0f,  0.99f, 1.0f, 1.0f,
        1.0f, -1.0f,  0.99f, 1.0f, 0.0f,
};
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
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
        glfwSetScrollCallback(mainWindow, scrollCallback);
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
        using namespace uf;
        auto vsS(fileToCharArr("./quad.vert"));
        auto fsS(fileToCharArr("./quad.frag"));
        renderQuad.program = makeProgram(compileShader(shDf(GL_VERTEX_SHADER,&vsS[0])),compileShader(shDf(GL_FRAGMENT_SHADER,&fsS[0])),true);
        renderQuad.texUniform = glGetUniformLocation(renderQuad.program,"tex");
    }
    //cuda buffer setup
    {
        size_t num_texels =  (size_t)width*height;
        size_t num_values = num_texels * 4;
        size_t size_tex_data = sizeof(GLubyte) * num_values;

        //TODO pinned memory
        checkCudaErrors(cudaMalloc((void **)&cudaDestResource, size_tex_data));
    }
    //TODO deletions here too

    size_t numVerts = 0,numSpheres = 0;
    //TODO move this to main and just copy to gpu here
    //cuda texture for triangles
    {
        using glm::vec4;


        TriMesh currentMesh(uf::loadTris("./cornell.obj"));
        thrust::host_vector<vec4> cpuTris1(currentMesh.ve);

        cout << "num verts: " << cpuTris1.size()<< endl;
        //TODO see if pinned memory here
        cudaMalloc(&gpuTris,sizeof(vec4)*cpuTris1.size());
        cudaMemcpy(gpuTris,thrust::raw_pointer_cast(&cpuTris1[0]),sizeof(vec4)*cpuTris1.size(),cudaMemcpyHostToDevice);


//TODO delete this code if not using textures anywhere else
//        trianglesTex.desc.resType = cudaResourceTypeLinear;
//        trianglesTex.desc.res.linear.devPtr = thrust::raw_pointer_cast(&gpuTris[0]);
//        trianglesTex.desc.res.linear.desc = cudaCreateChannelDesc<float4>();
//        trianglesTex.desc.res.linear.sizeInBytes = sizeof(float4)*cpuTris1.size();
//

//        memset(&trianglesTex.texDesc, 0, sizeof(trianglesTex.texDesc));
//        trianglesTex.textureObject = 0;
//        trianglesTex.texDesc.filterMode = cudaFilterModePoint;
//        trianglesTex.texDesc.normalizedCoords = 0;
//        trianglesTex.texDesc.addressMode[0] = cudaAddressModeWrap;

//        cudaCreateTextureObject(&trianglesTex.textureObject, &trianglesTex.desc, &trianglesTex.texDesc, NULL);

        numVerts = cpuTris1.size();
    }


    //load up spheres
    {

        using glm::vec4;
        //vec4 contains sphere pos and radius
        float rad= 30.0f;
        thrust::host_vector<Sphere> spheres;
        spheres.push_back(Sphere(vec4(0.0f, 0,-40,rad/2)));
        //TODO see if pinned memory here
        cudaMalloc(&gpuSpheres,sizeof(Sphere)*spheres.size());
        cudaMemcpy(gpuSpheres,thrust::raw_pointer_cast(&spheres[0]),sizeof(Sphere)*spheres.size(),cudaMemcpyHostToDevice);
        numSpheres = spheres.size();

    }
//    kernel default parameters
    {
        info.dev_drawRes = cudaDestResource;
        info.width = width;
        info.height = height;
        info.blockSize = dim3(16,16,1);
        info.triangleTex = gpuTris;
        info.numVerts = numVerts;
        info.sphereTex = gpuSpheres;
        info.numSpheres = numSpheres;
        info.cullBackFaces = true;
        info.depth = 1;
    }

    //setup camera
    {

        info.cam.dist = height/60;
        scrollSensitivity *= info.cam.dist;
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
        info.cam.pos   = glm::vec3(0,0,0);

    }


    //TODO remove this finally
    //cam debug
    {

        cout << glm::to_string(info.cam.up)<< endl;
        cout << glm::to_string(info.cam.right)<< endl;


//        objects need to have negative coords relative to camera
        const float xStep = (1/2.0f)*info.cam.dist*info.cam.aspect*info.cam.fov;
        const float yStep = (height - height/2.0f)*info.cam.dist*info.cam.fov/height;

        glm::vec3 dir = info.cam.front*info.cam.dist+info.cam.right*(1.0f*xStep)+info.cam.up*(1.0f*yStep);
        cout << "pos string "<<glm::to_string(dir)<< endl;
//    float3 dir = vtof3(cam.front*cam.dist+cam.right*(1.0f*xStep)+cam.up*(1.0f*yStep));



    }


    //TODO move this out later to a struct setup bvh
    {


        auto holdTris(uf::loadIndexedTris("./plane2.obj"));


        SceneMesh scene(holdTris.triIndexes.size(),holdTris.ve.size(),holdTris.triIndexes,holdTris.ve);



        for(int i = 0;i<holdTris.triIndexes.size();++i){
            cout << glm::to_string(holdTris.triIndexes[i].vertices) << endl;

        }
        Platform defaultplatform;
        BVH::BuildParams defaultparams;
        BVH::Stats stats;
        BVH myBVH(&scene, defaultplatform, defaultparams);


//        checkCudaErrors(cudaMalloc(&info.bvhData.dev_triindicesTpr,sizeof()));
//        checkCudaErrors(cudaMalloc());

//        cudaMalloc(info.bvhData.dev_triindicesTpr;

//        info.bvhData.dev_triNode;
//        info.bvhData.dev_triWoopTpr;






    }


}

BasicScene::~BasicScene() {






    checkCudaErrors(cudaGraphicsUnregisterResource(cudaTexResource));

    {
        glDeleteBuffers(1,&renderQuad.vbo);
        glDeleteVertexArrays(1,&renderQuad.vao);
    }

    {
        glDeleteProgram(renderQuad.program);
    }

    {
        checkCudaErrors(cudaFree(cudaDestResource));
    }

    //destroy triangles texture
    {
//        checkCudaErrors(cudaDestroyTextureObject(trianglesTex.textureObject));
        checkCudaErrors(cudaFree(gpuTris));
        checkCudaErrors(cudaFree(gpuSpheres));

    }

    glfwTerminate();
}

void BasicScene::run() {
    double delta = 0;
    double last = 0;
    glfwSetTime(last);
    uint64_t frameNumber = 0;
    while (!glfwWindowShouldClose(mainWindow)) {

        double curr = glfwGetTime();
        delta = curr-last;
        last = curr;

        glfwPollEvents();
        update(delta);

        //TODO see if lower level sync here
        //wait till prev frame done
        checkCudaErrors(cudaStreamSynchronize(0));

        info.hash = uf::hash(frameNumber);


        uf::GpuTimer g;
        g.Start();
        launchKernel(info);
        g.Stop();
        std::cout << g.Elapsed() << std::endl;



//        uf::GpuTimer memTimer;
//        memTimer.Start();
        cudaArray *texturePtr = nullptr;
        checkCudaErrors(cudaGraphicsMapResources(1, &cudaTexResource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texturePtr, cudaTexResource, 0, 0));

        size_t num_texels =  (size_t)width*height;
        size_t num_values = num_texels * 4;
        size_t size_tex_data = sizeof(GLubyte) * num_values;
        checkCudaErrors(cudaMemcpyToArray(texturePtr, 0, 0, cudaDestResource, size_tex_data, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTexResource, 0));
//        memTimer.Stop();
//        std::cout << memTimer.Elapsed() << std::endl;

        draw();

        glfwSwapBuffers(mainWindow);

        ++frameNumber;
    }





}



void mousePosCallback(GLFWwindow *window, double posX, double posY) {

}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    auto sn  = (BasicScene * )glfwGetWindowUserPointer(window);
    assert(sn!= nullptr && sn->mainWindow == window);

    //TODO imp add here the reset button to get back to camera default pos
    if(action == GLFW_PRESS) {
        if (key == GLFW_KEY_R) {

        }
        else if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
    }

}

void scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
    auto scn = (BasicScene *)glfwGetWindowUserPointer(window);
    assert(scn!= nullptr);
    using namespace std;
    float off = scrollSensitivity*(float)yoffset;
    cout << glm::to_string(scn->info.cam.front*off) << endl;
    scn->info.cam.pos+=off*scn->info.cam.front;








}

void BasicScene::Updater::operator()(double delta) {

    float camSpeed = moveSpeed*(float)delta;


    bool panning = false;

    if(glfwGetKey(prtScn.mainWindow,GLFW_KEY_W)){
        prtScn.info.cam.pos+=prtScn.info.cam.up*camSpeed;
    }
    if(glfwGetKey(prtScn.mainWindow,GLFW_KEY_S)){
        prtScn.info.cam.pos-=prtScn.info.cam.up*camSpeed;
    }
    if(glfwGetKey(prtScn.mainWindow,GLFW_KEY_A)){
        prtScn.info.cam.pos-=prtScn.info.cam.right*camSpeed;
    }
    if(glfwGetKey(prtScn.mainWindow,GLFW_KEY_D)){
        prtScn.info.cam.pos+=prtScn.info.cam.right*camSpeed;
    }
    if(glfwGetKey(prtScn.mainWindow,GLFW_KEY_LEFT_SHIFT)||glfwGetKey(prtScn.mainWindow,GLFW_KEY_RIGHT_SHIFT)){
        panning = true;
    }


    double xPos,yPos;
    glfwGetCursorPos(prtScn.mainWindow,&xPos,&yPos);
    if(!panning) {
        if (firstMouse) {
            firstMouse = false;
            lastX = xPos;
            lastY = yPos;
        }
        float offsetX = float(xPos - lastX);
        float offsetY = float(yPos - lastY);
        lastX = xPos;
        lastY = yPos;
        setPitchAndRoll(prtScn.info.cam, offsetX, offsetY);
    }
    else{
        //implement panning here


        firstMouse = true;
    }



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
