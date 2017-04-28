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
#include "CudaBVH.hpp"
#include <imgui.h>
#include <imgui_internal.h>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw_gl3.h>

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

    ImGui_ImplGlfwGL3_Init(mainWindow, true);
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




        checkCudaErrors(cudaMalloc(&info.accumBuffer,sizeof(vec3)*num_texels));
        checkCudaErrors(cudaMemset(info.accumBuffer,0,sizeof(vec3)*num_texels));

    }


    size_t numVerts = 0,numSpheres = 0;

    //cuda texture for triangles
    {
        using glm::vec4;

        TriMesh currentMesh(uf::loadTris("./sphere.obj"));

        thrust::host_vector<vec4> cpuTris1(currentMesh.ve);

//        thrust::host_vector<vec4> cpuTris1;
        cout << "num verts: " << cpuTris1.size()<< endl;
        //TODO see if pinned memory here
        cudaMalloc(&gpuTris,sizeof(vec4)*cpuTris1.size());
        cudaMemcpy(gpuTris,thrust::raw_pointer_cast(&cpuTris1[0]),sizeof(vec4)*cpuTris1.size(),cudaMemcpyHostToDevice);




        numVerts = cpuTris1.size();
    }


    //load up spheres
    {

        using glm::vec4;
        //vec4 contains sphere pos and radius
        float rad   = 600.0f;
        float pushX = 80;
        float pushY = 60;

        thrust::host_vector<Sphere> spheres;
        //posRad,emi,col
        spheres.push_back(Sphere(vec4(0.0f,-pushY-rad,-20,rad),vec3(0.0f,0.0f,0.0f),vec3(0.5f,0.5f,0.5f),Mat::DIFF));
        spheres.push_back(Sphere(vec4(0.0f, pushY+rad,-20,rad),vec3(0.0f,1.0f,0.0f),vec3(0.5f,0.5f,0.5f),Mat::DIFF));
        spheres.push_back(Sphere(vec4(-pushX-rad,0.0f,-20,rad),vec3(1.0f,0.0f,0.2f),vec3(0.5f,0.5f,0.5f),Mat::DIFF));
        spheres.push_back(Sphere(vec4( pushX+rad,0.0f,-20,rad),vec3(1.0f,1.0f,0.2f),vec3(0.5f,0.5f,0.5f),Mat::DIFF));
        spheres.push_back(Sphere(vec4( 0.0f,0.0f,-rad*1.5-20,rad),vec3(.0f,1.0f,0.8f),vec3(0.5f,0.5f,0.5f),Mat::DIFF));
        spheres.push_back(Sphere(vec4( 0.0f,0.0f,rad*1.5+20,rad),vec3(.0f,1.0f,0.8f),vec3(5.0f,5.0f,0.5f),Mat::DIFF));

        spheres.push_back(Sphere(vec4(0.0f, 0,-21,5),vec3(0.0f,0.0f,0.0f),vec3(1.0f,1.0f,1.0f),Mat::REFR));
        spheres.push_back(Sphere(vec4(12.0f, 0,-21,5),vec3(0.0f,0.0f,0.0f),vec3(1.0f,1.0f,1.0f),Mat::METAL));







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
        info.depth = 4;
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
        //save cam state to reset easily
        savecam = info.cam;

    }



    //cam debug
    {



//      objects need to have negative coords relative to camera
        const float xStep = (1/2.0f)*info.cam.dist*info.cam.aspect*info.cam.fov;
        const float yStep = (height - height/2.0f)*info.cam.dist*info.cam.fov/height;

        glm::vec3 dir = info.cam.front*info.cam.dist+info.cam.right*(1.0f*xStep)+info.cam.up*(1.0f*yStep);

    }



    {


        auto holdTris(uf::loadIndexedTris("./cornell_dragon.obj"));


        SceneMesh scene(holdTris.triIndexes.size(),holdTris.ve.size(),holdTris.triIndexes,holdTris.ve);


        Platform defaultplatform;
        BVH::BuildParams defaultparams;
        BVH::Stats stats;
        BVH myBVH(&scene, defaultplatform, defaultparams);

        std::cout << "Building CudaBVH\n";

        gpuBVH = new CudaBVH(myBVH,BVHLayout_Compact);



        // allocate and copy scene databuffers to the GPU (BVH nodes, triangle vertices, triangle indices)
        checkCudaErrors(cudaMalloc((void**)&info.bvhData.dev_triNode, gpuBVH->getGpuNodesSize() * sizeof(vec4)));
        checkCudaErrors(cudaMemcpy(info.bvhData.dev_triNode, gpuBVH->getGpuNodes(), gpuBVH->getGpuNodesSize() * sizeof(vec4), cudaMemcpyHostToDevice));


        checkCudaErrors(cudaMalloc((void**)&info.bvhData.dev_triIndicesPtr, gpuBVH->getGpuTriIndicesSize()* sizeof(int)));
        checkCudaErrors(cudaMemcpy(info.bvhData.dev_triIndicesPtr,gpuBVH->getGpuTriIndices(),gpuBVH->getGpuTriIndicesSize() * sizeof(int), cudaMemcpyHostToDevice));


        checkCudaErrors(cudaMalloc((void**)&info.bvhData.dev_triPtr, gpuBVH->getDebugTriSize()* sizeof(vec4)));
        checkCudaErrors(cudaMemcpy(info.bvhData.dev_triPtr,gpuBVH->getDebugTri(),gpuBVH->getDebugTriSize() * sizeof(vec4), cudaMemcpyHostToDevice));




        info.bvhData.triNodeSize = gpuBVH->getGpuNodesSize();
        info.bvhData.triIndicesSize = gpuBVH->getGpuTriIndicesSize();
        info.bvhData.triSize = gpuBVH->getDebugTriSize();




    }





}

BasicScene::~BasicScene() {






    free(gpuBVH);


    checkCudaErrors(cudaFree(info.bvhData.dev_triNode));
    checkCudaErrors(cudaFree(info.bvhData.dev_triIndicesPtr));
    checkCudaErrors(cudaFree(info.bvhData.dev_triPtr));


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


        //wait till prev frame done
        checkCudaErrors(cudaStreamSynchronize(0));

        info.hash = uf::hash(frameNumber);

        info.constantPdf =  (info.cam.dirty)?(1):(info.constantPdf+1);


        uf::GpuTimer g;
        g.Start();
        launchKernel(info);
        g.Stop();
        info.time_elapsed = g.Elapsed();
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

    if(action == GLFW_PRESS) {
        if (key == GLFW_KEY_R) {
            sn->info.cam = sn->savecam;
        }
        else if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
        else if(key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT){
            printf("pressed shift \n");
            sn->info.cam.enabled = !sn->info.cam.enabled;
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

    prtScn.info.cam.dirty = false;
    if(prtScn.info.cam.enabled) {


        if (glfwGetKey(prtScn.mainWindow, GLFW_KEY_W)) {
            prtScn.info.cam.dirty = true;
            prtScn.info.cam.pos += prtScn.info.cam.up * camSpeed;
        }
        if (glfwGetKey(prtScn.mainWindow, GLFW_KEY_S)) {
            prtScn.info.cam.dirty = true;
            prtScn.info.cam.pos -= prtScn.info.cam.up * camSpeed;
        }
        if (glfwGetKey(prtScn.mainWindow, GLFW_KEY_A)) {
            prtScn.info.cam.dirty = true;
            prtScn.info.cam.pos -= prtScn.info.cam.right * camSpeed;
        }
        if (glfwGetKey(prtScn.mainWindow, GLFW_KEY_D)) {
            prtScn.info.cam.dirty = true;
            prtScn.info.cam.pos += prtScn.info.cam.right * camSpeed;
        }


        double xPos, yPos;
        glfwGetCursorPos(prtScn.mainWindow, &xPos, &yPos);
        if (firstMouse) {
            firstMouse = false;
            lastX = xPos;
            lastY = yPos;
        }
        float offsetX = float(xPos - lastX);
        float offsetY = float(yPos - lastY);
        lastX = xPos;
        lastY = yPos;
        if (offsetX > prtScn.info.cam.camBias.x || offsetY > prtScn.info.cam.camBias.y)
            prtScn.info.cam.dirty = true;
        setPitchAndRoll(prtScn.info.cam, offsetX, offsetY);

    }
    else{
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

    glBindTexture(GL_TEXTURE_2D, renderQuad.tex);
    glActiveTexture(GL_TEXTURE0);

    glUniform1i(renderQuad.texUniform, 0);

    glBindVertexArray(renderQuad.vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);


    {
        ImGui_ImplGlfwGL3_NewFrame();
        drawWindow(true);
        ImGui::Render();
    }
}

void BasicScene::drawWindow(bool visible) {
    ImGui::Begin("Test Window", &visible);
    int b_size[2] = {(int)info.blockSize.x, (int)info.blockSize.y};

    ImGui::SliderInt2("BlockSize", b_size , 16, 256);
    ImGui::SliderInt("Depth", (int *) &info.depth, 1, 10);
    ImGui::SliderFloat("Air Reflective Index", &info.air_ref_index, 1.0f, 2.0f);
    ImGui::SliderFloat("Glass Reflective Index", &info.glass_ref_index, 1.0f, 2.0f);
    ImGui::Text("Time per frame: %0.2f ms", info.time_elapsed);
    ImGui::Checkbox("Back face culling", &info.cullBackFaces);
    //check box back face cull

    if (ImGui::Button("Button")) { std:: cout <<"button named button clicked" << std::endl;
    }
    info.blockSize.x = b_size[0];
    info.blockSize.y = b_size[1];
    ImGui::End();
}


