//
// Created by tgamerz on 21/4/17.
//

#ifndef GPUPROJECT_SCENEMESH_HPP
#define GPUPROJECT_SCENEMESH_HPP

#pragma once
#include <glm/vec3.hpp>
#include <vector>
#include <cassert>

class SceneMesh {
public:
    struct Triangle
    {
        glm::ivec3       vertices;   //3 vertex indices of triangle
        glm::vec3       normal;
        Triangle() : vertices(glm::ivec3(0, 0, 0)), normal(glm::vec3(0, 0, 0)) {}
    };


    thrust::host_vector<glm::vec3>         m_verts;
public:

    SceneMesh(const int numTris, const int numVerts, const std::vector<Triangle>& tris, const std::vector<glm::vec3>& verts) :
            m_numTris(numTris), m_numVerts(numVerts), m_tris(tris), m_verts(verts) {}

    ~SceneMesh(void);

    int             getNumTriangles(void) const   { return m_numTris; }
    const Triangle* getTrianglePtr(int idx = 0)   { assert(idx >= 0 && idx <= m_numTris); return (const Triangle*)&m_tris[0] + idx; }
    const Triangle& getTriangle(int idx)          { assert(idx < m_numTris); return m_tris[idx]; }

    int             getNumVertices(void) const    { return m_numVerts; }
    const glm::vec3*    getVertexPtr(int idx = 0)     { assert(idx >= 0 && idx <= m_numVerts); return (const glm::vec3*)&m_verts[0] + idx; }
    const glm::vec3&    getVertex(int idx)            { assert(idx < m_numVerts); return *getVertexPtr(idx); }

private:
    SceneMesh(const SceneMesh&); // forbidden
    SceneMesh&          operator=(const SceneMesh&); // forbidden

private:
    int             m_numTris;
    int             m_numVerts;
    std::vector<Triangle>      m_tris;
};


#endif //GPUPROJECT_SCENEMESH_HPP
