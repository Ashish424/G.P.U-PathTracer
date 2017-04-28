/*
*  Copyright (c) 2009-2011, NVIDIA Corporation
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*      * Redistributions of source code must retain the above copyright
*        notice, this list of conditions and the following disclaimer.
*      * Redistributions in binary form must reproduce the above copyright
*        notice, this list of conditions and the following disclaimer in the
*        documentation and/or other materials provided with the distribution.
*      * Neither the name of NVIDIA Corporation nor the
*        names of its contributors may be used to endorse or promote products
*        derived from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
*  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
*  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
