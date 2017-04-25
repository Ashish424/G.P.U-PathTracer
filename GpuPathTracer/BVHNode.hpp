//
// Created by tgamerz on 21/4/17.
//

#ifndef GPUPROJECT_BVHNODE_HPP
#define GPUPROJECT_BVHNODE_HPP

#include <glm/vec3.hpp>
#include "bvh_util.hpp"
#include <vector>
#include <cassert>

enum BVH_STAT
{
    BVH_STAT_NODE_COUNT,
    BVH_STAT_INNER_COUNT,
    BVH_STAT_LEAF_COUNT,
    BVH_STAT_TRIANGLE_COUNT,
    BVH_STAT_CHILDNODE_COUNT,
};

class AABB
{
public:
    typedef glm::vec3 vec3;
    inline                    AABB(void) : m_mn(FW_F32_MAX, FW_F32_MAX, FW_F32_MAX), m_mx(-FW_F32_MAX, -FW_F32_MAX, -FW_F32_MAX) {}
    inline                    AABB(const vec3& mn, const vec3& mx) : m_mn(mn), m_mx(mx) {}

    inline    void            grow(const vec3& pt)   { m_mn = min3f(m_mn, pt); m_mx = max3f(m_mx, pt); } // grows bounds to include 3d point pt
    inline    void            grow(const AABB& aabb)  { grow(aabb.m_mn); grow(aabb.m_mx); }
    inline    void            intersect(const AABB& aabb)  { m_mn = max3f(m_mn, aabb.m_mn); m_mx = min3f(m_mx, aabb.m_mx); } /// box formed by intersection of 2 AABB boxes
    inline    float           volume(void) const      { if (!valid()) return 0.0f; return (m_mx.x - m_mn.x) * (m_mx.y - m_mn.y) * (m_mx.z - m_mn.z); } /// volume = AABB side along X-axis * side along Y * side along Z
    inline    float           area(void) const        { if (!valid()) return 0.0f; vec3 d = m_mx - m_mn; return (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f; }
    inline    bool            valid(void) const       { return m_mn.x <= m_mx.x && m_mn.y <= m_mx.y && m_mn.z <= m_mx.z; }
    inline    vec3           midPoint(void) const    { return (m_mn + m_mx)*0.5f; } // AABB centroid or midpoint
    inline    const vec3&    min(void) const         { return m_mn; }
    inline    const vec3&    max(void) const         { return m_mx; }
    inline    vec3&          min(void)               { return m_mn; }
    inline    vec3&          max(void)               { return m_mx; }

    inline    AABB            operator+   (const AABB& aabb) const { AABB u(*this); u.grow(aabb); return u; }

private:
    vec3           m_mn; // AABB min bound
    vec3           m_mx; // AABB max bound
};

class BVHNode {
public:
    BVHNode() : m_probability(1.f), m_parentProbability(1.f), m_treelet(-1), m_index(-1) {}
    virtual bool        isLeaf() const = 0;
    virtual int         getNumChildNodes() const = 0;
    virtual BVHNode*    getChildNode(int i) const = 0;
    virtual int         getNumTriangles() const { return 0; }

    float       getArea() const     { return m_bounds.area(); }

    AABB        m_bounds;

    // These are somewhat experimental, for some specific test and may be invalid...
    float       m_probability;          // probability of coming here (widebvh uses this)
    float       m_parentProbability;    // probability of coming to parent (widebvh uses this)

    int         m_treelet;              // for queuing tests (qmachine uses this)
    int         m_index;                // in linearized tree (qmachine uses this)

    // Subtree functions
    int     getSubtreeSize(BVH_STAT stat = BVH_STAT_NODE_COUNT) const;
    void    computeSubtreeProbabilities(const Platform& p, float parentProbability, float& sah);
    float   computeSubtreeSAHCost(const Platform& p) const;     // NOTE: assumes valid probabilities
    void    deleteSubtree();

    void    assignIndicesDepthFirst(int index = 0, bool includeLeafNodes = true);
    void    assignIndicesBreadthFirst(int index = 0, bool includeLeafNodes = true);

};

class InnerNode : public BVHNode
{
public:
    InnerNode(const AABB& bounds, BVHNode* child0, BVHNode* child1)   { m_bounds = bounds; m_children[0] = child0; m_children[1] = child1; }

    bool        isLeaf() const                  { return false; }
    int         getNumChildNodes() const        { return 2; }
    BVHNode*    getChildNode(int i) const       { assert(i >= 0 && i<2); return m_children[i]; }

    BVHNode*    m_children[2];
};


class LeafNode : public BVHNode
{
public:
    LeafNode(const AABB& bounds, int lo, int hi)  { m_bounds = bounds; m_lo = lo; m_hi = hi; }
    LeafNode(const LeafNode& s)                 { *this = s; }

    bool        isLeaf() const                  { return true; }
    int         getNumChildNodes() const        { return 0; }  // leafnode has 0 children
    BVHNode*    getChildNode(int) const         { return NULL; }

    int         getNumTriangles() const         { return m_hi - m_lo; }
    int         m_lo;  // lower index in triangle list
    int         m_hi;  // higher index in triangle list
};

#endif //GPUPROJECT_BVHNODE_HPP
