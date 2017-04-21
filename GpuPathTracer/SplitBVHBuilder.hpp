//
// Created by tgamerz on 21/4/17.
//

#ifndef GPUPROJECT_SPLITBVHBUILDER_HPP
#define GPUPROJECT_SPLITBVHBUILDER_HPP

#pragma once
#include "BVH.hpp"

class SplitBVHBuilder {
private:
    enum
    {
        MaxDepth = 64,
        MaxSpatialDepth = 48,
        NumSpatialBins = 32,
    };

    struct Reference   /// a AABB bounding box enclosing 1 triangle, a reference can be duplicated by a split to be contained in 2 AABB boxes
    {
        int                 triIdx;
        AABB                bounds;

        Reference(void) : triIdx(-1) {}  /// constructor
    };

    struct NodeSpec
    {
        int                 numRef;   // number of references contained by node
        AABB                bounds;

        NodeSpec(void) : numRef(0) {}
    };

    struct ObjectSplit
    {
        float                 sah;   // cost
        int                 sortDim;  // axis along which triangles are sorted
        int                 numLeft;  // number of triangles (references) in left child
        AABB                leftBounds;
        AABB                rightBounds;

        ObjectSplit(void) : sah(FW_F32_MAX), sortDim(0), numLeft(0) {}
    };

    struct SpatialSplit
    {
        float                 sah;
        int                 dim;   /// split axis
        float                 pos;   /// position of split along axis (dim)

        SpatialSplit(void) : sah(FW_F32_MAX), dim(0), pos(0.0f) {}
    };

    struct SpatialBin
    {
        AABB                bounds;
        int                 enter;
        int                 exit;
    };

public:
    SplitBVHBuilder(BVH& bvh, const BVH::BuildParams& params);
    ~SplitBVHBuilder(void);

    BVHNode*                run(void);

private:
    static int              sortCompare(void* data, int idxA, int idxB);
    static void             sortSwap(void* data, int idxA, int idxB);

    BVHNode*                buildNode(const NodeSpec& spec, int level, float progressStart, float progressEnd);
    BVHNode*                createLeaf(const NodeSpec& spec);

    ObjectSplit             findObjectSplit(const NodeSpec& spec, float nodeSAH);
    void                    performObjectSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split);

    SpatialSplit            findSpatialSplit(const NodeSpec& spec, float nodeSAH);
    void                    performSpatialSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SpatialSplit& split);
    void                    splitReference(Reference& left, Reference& right, const Reference& ref, int dim, float pos);

private:
    SplitBVHBuilder(const SplitBVHBuilder&); // forbidden
    SplitBVHBuilder&        operator=           (const SplitBVHBuilder&); // forbidden

private:
    BVH&                    m_bvh;
    const Platform&         m_platform;
    const BVH::BuildParams& m_params;

    std::vector<Reference>      m_refStack;
    float                     m_minOverlap;
    std::vector<AABB>           m_rightBounds;
    int                     m_sortDim;
    SpatialBin              m_bins[3][NumSpatialBins];

    //Timer                   m_progressTimer;
    int                     m_numDuplicates;

};


#endif //GPUPROJECT_SPLITBVHBUILDER_HPP
