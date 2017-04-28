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
    bool dirty = true;
    glm::vec2 camBias = glm::vec2(0.1,0.1);
    bool enabled = true;
};


struct BVHData{


    vec4 * dev_triNode = nullptr;
    size_t triNodeSize;
    int * dev_triIndicesPtr = nullptr;
    size_t triIndicesSize;


    vec4 *dev_triPtr;
    size_t triSize;
};


struct kernelInfo{
    dim3 blockSize;
    uint64_t hash = 0;
    unsigned int * dev_drawRes;
    vec3 * accumBuffer = nullptr;

    int width,height;
    CamInfo cam;
    glm::vec4 *  triangleTex = nullptr;
    size_t numVerts = 0;
    Sphere *  sphereTex = nullptr;
    size_t numSpheres = 0;
    bool cullBackFaces = true;
    unsigned int depth = 1;


    uint64_t constantPdf = 1;
    BVHData bvhData;

    float time_elapsed;
    float air_ref_index = 1.0f;
    float glass_ref_index = 1.4f;
};


#endif //GPUPROJECT_CPUSTRUCTS_HPP
