//
// Created by ashish on 4/25/17.
//

#ifndef GPUPROJECT_CPUSTRUCTS_HPP
#define GPUPROJECT_CPUSTRUCTS_HPP
struct tex1DInfo{
    void                   *h_data;
    cudaResourceDesc desc;
    cudaTextureDesc texDesc;
    cudaTextureObject_t     textureObject;

    tex1DInfo()
    {
        memset(this,0,sizeof(tex1DInfo));
    }

};
struct CamInfo{
    glm::vec3 front,right,up,pos;
    float dist;
    float pitch ,yaw;
    float aspect;
    float fov;


};


struct BVHData{


    vec4 * dev_triNode = nullptr;
    size_t triNodeSize;
    vec4 * dev_triWoopTpr = nullptr;
    size_t triWoopSize;
    int * dev_triIndicesTpr = nullptr;
    size_t triIndicesSize;

    size_t leafCount;
    size_t triCount;
    vec4 *dev_triDebugPtr;
    size_t triDebugSize;
};


struct kernelInfo{
    dim3 blockSize;
    uint64_t hash = 0;
    unsigned int * dev_drawRes;
    int width,height;
    CamInfo cam;
    glm::vec4 *  triangleTex = nullptr;
    size_t numVerts = 0;
    Sphere *  sphereTex = nullptr;
    size_t numSpheres = 0;
    bool cullBackFaces = true;
    unsigned int depth = 1;



    BVHData bvhData;

};


#endif //GPUPROJECT_CPUSTRUCTS_HPP
