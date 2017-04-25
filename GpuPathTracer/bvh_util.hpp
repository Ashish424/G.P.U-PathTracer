//
// Created by tgamerz on 21/4/17.
//

#ifndef GPUPROJECT_BVH_UTIL_HPP
#define GPUPROJECT_BVH_UTIL_HPP

#include <string>
#include <cuda_runtime.h>
#include <glm/vec3.hpp>
#define FW_F32_MIN          (1.175494351e-38f)
#define FW_F32_MAX          (3.402823466e+38f)

using glm::vec3;

enum BVHLayout
{
    BVHLayout_AOS_AOS = 0,              // Nodes = array-of-structures, triangles = array-of-structures. Used by tesla_xxx kernels.
    BVHLayout_AOS_SOA,                  // Nodes = array-of-structures, triangles = structure-of-arrays.
    BVHLayout_SOA_AOS,                  // Nodes = structure-of-arrays, triangles = array-of-structures.
    BVHLayout_SOA_SOA,                  // Nodes = structure-of-arrays, triangles = structure-of-arrays.
    BVHLayout_Compact,                  // Variant of BVHLayout_AOS_AOS with implicit leaf nodes.
    BVHLayout_Compact2,                 // Variant of BVHLayout_AOS_AOS with implicit leaf nodes.

    BVHLayout_Max
};

unsigned int WangHash(unsigned int a);

inline __host__ __device__ vec3 min3f(const vec3& v1, const vec3& v2){ return vec3(v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y, v1.z < v2.z ? v1.z : v2.z); }
inline __host__ __device__ vec3 max3f(const vec3& v1, const vec3& v2){ return vec3(v1.x > v2.x ? v1.x : v2.x, v1.y > v2.y ? v1.y : v2.y, v1.z > v2.z ? v1.z : v2.z); }

class Platform
{
public:
    Platform() { m_name = std::string("Default"); m_SAHNodeCost = 1.f; m_SAHTriangleCost = 1.f; m_nodeBatchSize = 1; m_triBatchSize = 1; m_minLeafSize = 1; m_maxLeafSize = 0x7FFFFFF; } /// leafsize = aantal tris
    Platform(const std::string& name, float nodeCost = 1.f, float triCost = 1.f, int nodeBatchSize = 1, int triBatchSize = 1) { m_name = name; m_SAHNodeCost = nodeCost; m_SAHTriangleCost = triCost; m_nodeBatchSize = nodeBatchSize; m_triBatchSize = triBatchSize; m_minLeafSize = 1; m_maxLeafSize = 0x7FFFFFF; }

    const std::string&   getName() const                { return m_name; }

    // SAH weights
    float getSAHTriangleCost() const                    { return m_SAHTriangleCost; }
    float getSAHNodeCost() const                        { return m_SAHNodeCost; }

    // SAH costs, raw and batched
    float getCost(int numChildNodes, int numTris) const { return getNodeCost(numChildNodes) + getTriangleCost(numTris); }
    float getTriangleCost(int n) const                  { return roundToTriangleBatchSize(n) * m_SAHTriangleCost; }
    float getNodeCost(int n) const                      { return roundToNodeBatchSize(n) * m_SAHNodeCost; }


    // batch processing (how many ops at the price of one)
    int   getTriangleBatchSize() const                  { return m_triBatchSize; }
    int   getNodeBatchSize() const                      { return m_nodeBatchSize; }
    void  setTriangleBatchSize(int triBatchSize)        { m_triBatchSize = triBatchSize; }
    void  setNodeBatchSize(int nodeBatchSize)           { m_nodeBatchSize = nodeBatchSize; }
    int   roundToTriangleBatchSize(int n) const         { return ((n + m_triBatchSize - 1) / m_triBatchSize)*m_triBatchSize; }
    int   roundToNodeBatchSize(int n) const             { return ((n + m_nodeBatchSize - 1) / m_nodeBatchSize)*m_nodeBatchSize; }


    // leaf preferences
    void  setLeafPreferences(int minSize, int maxSize)   { m_minLeafSize = minSize; m_maxLeafSize = maxSize; }
    int   getMinLeafSize() const                        { return m_minLeafSize; }
    int   getMaxLeafSize() const                        { return m_maxLeafSize; }

private:
    std::string  m_name;
    float   m_SAHNodeCost;
    float   m_SAHTriangleCost;
    int     m_triBatchSize;
    int     m_nodeBatchSize;
    int     m_minLeafSize;
    int     m_maxLeafSize;
};




#endif //GPUPROJECT_BVH_UTIL_HPP
