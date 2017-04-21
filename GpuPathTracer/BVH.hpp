//
// Created by tgamerz on 21/4/17.
//

#ifndef GPUPROJECT_BVH_HPP
#define GPUPROJECT_BVH_HPP

#include "BVHNode.hpp"
#include "utilfun.hpp"
#include "SceneMesh.hpp"
#include <cstdio>
#include <string>
#include <cstring>

class BVH {
public:
    struct Stats
    {
        Stats()             { clear(); }
        void clear()        { memset(this, 0, sizeof(Stats)); }
        void print() const  {} //printf("Tree stats: [bfactor=%d] %d nodes (%d+%d), %.2f SAHCost, %.1f children/inner, %.1f tris/leaf\n", branchingFactor, numLeafNodes + numInnerNodes, numLeafNodes, numInnerNodes, SAHCost, 1.f*numChildNodes / max1i(numInnerNodes, 1), 1.f*numTris / max1i(numLeafNodes, 1)); }

        float     SAHCost;           // Surface Area Heuristic cost
        int     branchingFactor;
        int     numInnerNodes;
        int     numLeafNodes;
        int     numChildNodes;
        int     numTris;
    };

    struct BuildParams
    {
        Stats*      stats;
        bool        enablePrints;
        float         splitAlpha;     // spatial split area threshold, see Nvidia paper on SBVH by Martin Stich, usually 0.05

        BuildParams(void)
        {
            stats = NULL;
            enablePrints = true;
            splitAlpha = 1.0e-5f;
        }

    };

public:
    BVH(SceneMesh* meshes, const Platform& platform, const BuildParams& params);
    ~BVH(void)                  { if (m_root) m_root->deleteSubtree(); }

    SceneMesh*              getSceneMesh(void) const           { return m_meshes; }
    const Platform&     getPlatform(void) const        { return m_platform; }
    BVHNode*            getRoot(void) const            { return m_root; }

    std::vector<int>&         getTriIndices(void)                  { return m_triIndices; }
    const std::vector<int>&   getTriIndices(void) const            { return m_triIndices; }

private:

    SceneMesh*              m_meshes;
    Platform            m_platform;

    BVHNode*            m_root;
    std::vector<int>        m_triIndices;
};


#endif //GPUPROJECT_BVH_HPP
