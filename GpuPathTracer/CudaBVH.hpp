//
// Created by tgamerz on 22/4/17.
//

#ifndef GPUPROJECT_CUDABVH_HPP
#define GPUPROJECT_CUDABVH_HPP

#include "BVH.hpp"
#include "bvh_util.hpp"
#include <thrust/host_vector.h>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

using glm::ivec4;
using glm::vec4;
using glm::ivec2;

class CudaBVH {
public:
    enum
    {
        Align = 4096
    };

public:
    explicit    CudaBVH(const BVH& bvh, BVHLayout layout);
    CudaBVH(CudaBVH& other)        { operator=(other); }
    ~CudaBVH(void);

    BVHLayout   getLayout(void) const            { return m_layout; }
    thrust::host_vector<ivec4>&  getNodeBuffer(void)            { return m_nodes; }
    thrust::host_vector<ivec4>&  getTriWoopBuffer(void)         { return m_triWoop; }
    thrust::host_vector<int>&    getTriIndexBuffer(void)        { return m_triIndex; }

    ivec4*  getGpuNodes(void)            { return m_gpuNodes; }
    ivec4*  getGpuTriWoop(void)         { return m_gpuTriWoop; }
    ivec4*  getDebugTri(void)			{ return m_debugTri;  }
    int*    getGpuTriIndices(void)        { return m_gpuTriIndices; }

    unsigned int    getGpuNodesSize(void)			{ return m_gpuNodesSize; }
    unsigned int    getGpuTriWoopSize(void)			{ return m_gpuTriWoopSize; }
    unsigned int    getDebugTriSize(void)			{ return m_debugTriSize; }
    unsigned int    getGpuTriIndicesSize(void)        { return m_gpuTriIndicesSize; }
    unsigned int    getLeafnodeCount(void)			{ return m_leafnodecount; }
    unsigned int    getTriCount(void)			{ return m_tricount; }

    // AOS: idx ignored, returns entire buffer
    // SOA: 0 <= idx < 4, returns one subarray  // idx between 0 and 4
    ivec2       getNodeSubArray(int idx) const; // (ofs, size)
    ivec2       getTriWoopSubArray(int idx) const; // (ofs, size)

    CudaBVH&    operator=(CudaBVH& other);

private:
    void        createNodeBasic(const BVH& bvh);
    void        createTriWoopBasic(const BVH& bvh);
    void        createTriIndexBasic(const BVH& bvh);
    void        createCompact(const BVH& bvh, int nodeOffsetSizeDiv);
    void        woopifyTri(const BVH& bvh, int idx);

private:
    BVHLayout   m_layout;

    thrust::host_vector<ivec4>      m_nodes;
    thrust::host_vector<ivec4>      m_triWoop;
    thrust::host_vector<int>        m_triIndex;

    ivec4*	m_gpuNodes;
    ivec4*  m_gpuTriWoop;
    ivec4*  m_debugTri;
    int*	m_gpuTriIndices;

    unsigned int     m_gpuNodesSize;
    unsigned int		m_gpuTriWoopSize;
    unsigned int     m_debugTriSize;
    unsigned int		m_gpuTriIndicesSize;
    unsigned int		m_leafnodecount;
    unsigned int     m_tricount;

    vec4   m_woop[3];
    vec4	m_debugtri[3];

};


#endif //GPUPROJECT_CUDABVH_HPP
