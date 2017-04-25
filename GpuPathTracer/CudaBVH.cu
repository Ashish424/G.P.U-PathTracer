//
// Created by tgamerz on 22/4/17.
//

#include "CudaBVH.hpp"
#include <cassert>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
using glm::fmat4;
using glm::ivec3;
using glm::vec3;

static int woopcount = 0;

template <typename T>
inline void addPtrToVec(thrust::host_vector<T> &arr, T *ptr, int size){
    if(!size)return;
    else {
        for(int i = 0; i < size; i++){
            arr.push_back(ptr[i]);
        }
    }
}

inline unsigned int          floatToBits(float a)         { return *(unsigned int *)&a; }

CudaBVH::CudaBVH(const BVH& bvh, BVHLayout layout)
        : m_layout(layout)
{
    assert(layout >= 0 && layout < BVHLayout_Max);

    if (layout == BVHLayout_Compact)
    {
        createCompact(bvh, 1);
        return;
    }

    if (layout == BVHLayout_Compact2)
    {
        createCompact(bvh, 16);
        return;
    }
}

CudaBVH::~CudaBVH(void)
{
}

//------------------------------------------------------------------------

ivec2 CudaBVH::getNodeSubArray(int idx) const
{
    assert(idx >= 0 && idx < 4);
    int size = (int)m_nodes.size();

    if (m_layout == BVHLayout_SOA_AOS || m_layout == BVHLayout_SOA_SOA)
        return ivec2((size >> 2) * idx, (size >> 2));
    return ivec2(0, size);
}

//------------------------------------------------------------------------

ivec2 CudaBVH::getTriWoopSubArray(int idx) const
{
    assert(idx >= 0 && idx < 4);
    int size = (int)m_triWoop.size();

    if (m_layout == BVHLayout_AOS_SOA || m_layout == BVHLayout_SOA_SOA)
        return ivec2((size >> 2) * idx, (size >> 2));
    return ivec2(0, size);
}

//------------------------------------------------------------------------

CudaBVH& CudaBVH::operator=(CudaBVH& other)
{
    if (&other != this)
    {
        m_layout = other.m_layout;
        m_nodes = other.m_nodes;
        m_triWoop = other.m_triWoop;
        m_triIndex = other.m_triIndex;
    }
    return *this;
}

namespace detail
{
    struct StackEntry
    {
        const BVHNode*  node;
        int             idx;

        StackEntry(const BVHNode* n = NULL, int i = 0) : node(n), idx(i) {}
    };
}

void CudaBVH::createCompact(const BVH& bvh, int nodeOffsetSizeDiv)
{
    using namespace detail; // for StackEntry

    int leafcount = 0; // counts leafnodes

    // construct and initialize data arrays which will be copied to CudaBVH buffers (last part of this function).
    //TODO:  Taken assumption
    thrust::host_vector<ivec4> nodeData(4);
    thrust::host_vector<ivec4> triWoopData;
    thrust::host_vector<ivec4> triDebugData; // array for regular (non-woop) triangles
    thrust::host_vector<int> triIndexData;

    // construct a stack (array of stack entries) to help in filling the data arrays
    thrust::host_vector<StackEntry> stack;
    stack.push_back(StackEntry(bvh.getRoot(), 0));// initialise stack to rootnode

    while (stack.size()) // while stack is not empty
    {
        StackEntry e = stack.back(); stack.pop_back(); // pop the stack
        assert(e.node->getNumChildNodes() == 2);
        const AABB* cbox[2];
        int cidx[2]; // stores indices to both children

        // Process children.

        // for each child in entry e
        for (int i = 0; i < 2; i++)
        {
            const BVHNode* child = e.node->getChildNode(i); // current childnode
            cbox[i] = &child->m_bounds; // current child's AABB

            ////////////////////////////
            /// INNER NODE
            //////////////////////////////

            // Inner node => push to stack.

            if (!child->isLeaf()) // no leaf, thus an inner node
            {   // compute childindex
                cidx[i] = (nodeData.size() * sizeof(ivec4)) / nodeOffsetSizeDiv; // nodeOffsetSizeDiv is 1 for Fermi kernel, 16 for Kepler kernel

                // push the current child on the stack
                stack.push_back(StackEntry(child, nodeData.size()));
                //TODO Taken assumption here
                nodeData.push_back(ivec4(0, 0, 0, 0));
                nodeData.push_back(ivec4(0, 0, 0, 0));
                nodeData.push_back(ivec4(0, 0, 0, 0));
                nodeData.push_back(ivec4(0, 0, 0, 0));/// adds 4 * ivec4 per inner node or 4 * 16 bytes/ivec4 = 64 bytes of empty data per inner node
                continue; // process remaining childnode (if any)
            }



            //////////////////////
            /// LEAF NODE
            /////////////////////

            // Leaf => append triangles.

            const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child);

            // index of a leafnode is a negative number, hence the ~
            cidx[i] = ~((int)triWoopData.size());  // leafs must be stored as negative (bitwise complement) in order to be recognised by pathtracer as a leaf

            // for each triangle in leaf, range of triangle index j from m_lo to m_hi
            for (int j = leaf->m_lo; j < leaf->m_hi; j++)
            {
                // transform the triangle's vertices to Woop triangle (simple transform to right angled triangle, see paper by Sven Woop)
                woopifyTri(bvh, j);  /// j is de triangle index in triIndex array

                if (m_woop[0].x == 0.0f) m_woop[0].x = 0.0f;  // avoid degenerate coordinates
                // add the transformed woop triangle to triWoopData

//                triWoopData.add((ivec4*)m_woop, 3);
                addPtrToVec(triWoopData, (ivec4*)m_woop, 3);
//                triDebugData.add((ivec4*)m_debugtri, 3);
                addPtrToVec(triDebugData, (ivec4*)m_debugtri, 3);

                // add tri index for current triangle to triIndexData
                triIndexData.push_back(bvh.getTriIndices()[j]);
                triIndexData.push_back(0); // zero padding because CUDA kernel uses same index for vertex array (3 vertices per triangle)
                triIndexData.push_back(0); // and array of triangle indices
            }

            // Leaf node terminator to indicate end of leaf, stores hexadecimal value 0x80000000 (= 2147483648 in decimal)
            //TODO : Beware Big assumption here
            triWoopData.push_back(ivec4(0x80000000,0x80000000,0x80000000,0x80000000)); // leafnode terminator code indicates the last triangle of the leaf node
            triDebugData.push_back(ivec4(0x80000000,0x80000000,0x80000000,0x80000000));

            // add extra zero to triangle indices array to indicate end of leaf
            triIndexData.push_back(0);  // terminates triIndexdata for current leaf

            leafcount++;
        }

        // Write entry for current node.
        /// 4 ivec4 per node (according to compact bvh node layout)
        ivec4* dst = &nodeData[e.idx];
        ///std::cout << "e.idx: " << e.idx << " cidx[0]: " << cidx[0] << " cidx[1]: " << cidx[1] << "\n";
        dst[0] = ivec4(floatToBits(cbox[0]->min().x), floatToBits(cbox[0]->max().x), floatToBits(cbox[0]->min().y), floatToBits(cbox[0]->max().y));
        dst[1] = ivec4(floatToBits(cbox[1]->min().x), floatToBits(cbox[1]->max().x), floatToBits(cbox[1]->min().y), floatToBits(cbox[1]->max().y));
        dst[2] = ivec4(floatToBits(cbox[0]->min().z), floatToBits(cbox[0]->max().z), floatToBits(cbox[1]->min().z), floatToBits(cbox[1]->max().z));
        dst[3] = ivec4(cidx[0], cidx[1], 0, 0);

    } // end of while loop, will iteratively empty the stack


    m_leafnodecount = leafcount;
    m_tricount = woopcount;

    // Write data arrays to arrays of CudaBVH

    m_gpuNodes = (ivec4*) malloc(nodeData.size() * sizeof(ivec4));
    m_gpuNodesSize = nodeData.size();

    for (int i = 0; i < nodeData.size(); i++){
        m_gpuNodes[i].x = nodeData[i].x;
        m_gpuNodes[i].y = nodeData[i].y;
        m_gpuNodes[i].z = nodeData[i].z;
        m_gpuNodes[i].w = nodeData[i].w; // child indices
    }

    m_gpuTriWoop = (ivec4*) malloc(triWoopData.size() * sizeof(ivec4));
    m_gpuTriWoopSize = triWoopData.size();

    for (int i = 0; i < triWoopData.size(); i++){
        m_gpuTriWoop[i].x = triWoopData[i].x;
        m_gpuTriWoop[i].y = triWoopData[i].y;
        m_gpuTriWoop[i].z = triWoopData[i].z;
        m_gpuTriWoop[i].w = triWoopData[i].w;
    }

    m_debugTri = (ivec4*)malloc(triDebugData.size() * sizeof(ivec4));
    m_debugTriSize = triDebugData.size();

    for (int i = 0; i < triDebugData.size(); i++){
        m_debugTri[i].x = triDebugData[i].x;
        m_debugTri[i].y = triDebugData[i].y;
        m_debugTri[i].z = triDebugData[i].z;
        m_debugTri[i].w = triDebugData[i].w;
    }

    m_gpuTriIndices = (int*) malloc(triIndexData.size() * sizeof(int));
    m_gpuTriIndicesSize = triIndexData.size();

    for (int i = 0; i < triIndexData.size(); i++){
        m_gpuTriIndices[i] = triIndexData[i];
    }
}

//------------------------------------------------------------------------

void CudaBVH::woopifyTri(const BVH& bvh, int triIdx)
{
    woopcount++;

    // fetch the 3 vertex indices of this triangle
    const ivec3& vtxInds = bvh.getSceneMesh()->getTriangle(bvh.getTriIndices()[triIdx]).vertices;
    thrust::host_vector<vec3> &vertices = bvh.getSceneMesh()->m_verts;
    const vec3& v0 = vec3(vertices[vtxInds[0]].x, vertices[vtxInds[0]].y, vertices[vtxInds[0]].z); // vtx xyz pos voor eerste triangle vtx
    //const vec3& v1 = bvh.getScene()->getVertex(vtxInds.y);
    const vec3& v1 = vec3(vertices[vtxInds[1]].x, vertices[vtxInds[1]].y, vertices[vtxInds[1]].z); // vtx xyz pos voor tweede triangle vtx
    //const vec3& v2 = bvh.getScene()->getVertex(vtxInds.z);
    const vec3& v2 = vec3(vertices[vtxInds[2]].x, vertices[vtxInds[2]].y, vertices[vtxInds[2]].z); // vtx xyz pos voor derde triangle vtx

    // regular triangles (for debugging only)
    m_debugtri[0] = vec4(v0.x, v0.y, v0.z, 0.0f);
    m_debugtri[1] = vec4(v1.x, v1.y, v1.z, 0.0f);
    m_debugtri[2] = vec4(v2.x, v2.y, v2.z, 0.0f);

    glm::mat4 mtx;
    // compute edges and transform them with a matrix
    mtx[0] = glm::vec4(v0 - v2, 0.0f);
    mtx[1] = glm::vec4(v1 - v2, 0.0f);
    mtx[2] = glm::vec4(cross(v0 - v2, v1 - v2), 0.0f);
    mtx[3] = glm::vec4(v0 - v2, 1.0f);

    mtx = glm::inverse(mtx);

    /// m_woop[3] stores 3 transformed triangle edges
    m_woop[0] = vec4(mtx[0][2], mtx[1][2], mtx[2][2], -mtx[3][2]); // elements of 3rd row of inverted matrix
    m_woop[1] = vec4(mtx[0][0], mtx[1][0], mtx[2][0], mtx[3][0]);
    m_woop[2] = vec4(mtx[0][1], mtx[1][1], mtx[2][1], mtx[3][1]);
}
