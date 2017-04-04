#include <vector>


#include "ray.hpp"
#include "triangle.hpp"
#include "kdtree.h"
KDNode* KDNode::build(std::vector<Triangle*> &tris, int depth){
    KDNode* node = new KDNode();
    node->leaf = false;node->triangles = std::vector<Triangle*>();node->left = NULL;node->right = NULL;
    node->box = AABBox();if (tris.size() == 0) return node;
    if (depth > 3*25 || tris.size() <= 6) {
        node->triangles = tris;
        node->leaf = true;
        node->box = tris[0]->getBoundingBox();

        for (long i=1; i<tris.size(); i++)
            node->box.expand(tris[i]->getBoundingBox());

        node->left = new KDNode();node->right = new KDNode();
        node->left->triangles = std::vector<Triangle*>();
        node->right->triangles = std::vector<Triangle*>();

        return node;
    }

    node->box = tris[0]->getBoundingBox();
    glm::dvec3 midpt = glm::dvec3();
    double tris_recp = 1.0/tris.size();

    for (long i=1; i<tris.size(); i++) {
        node->box.expand(tris[i]->getBoundingBox());
        midpt = midpt + (tris[i]->getMidpoint() * tris_recp);
    }

    std::vector<Triangle*> leftTris,rightTris;
    int axis = node->box.get_longest_axis();
    for (long i=0; i<tris.size(); i++) {
        switch (axis) {
            case 0:midpt.x >= tris[i]->getMidpoint().x?rightTris.push_back(tris[i]):leftTris.push_back(tris[i]);
                break;
            case 1:midpt.y >= tris[i]->getMidpoint().y?rightTris.push_back(tris[i]):leftTris.push_back(tris[i]);
                break;
            case 2:midpt.z >= tris[i]->getMidpoint().z?rightTris.push_back(tris[i]):leftTris.push_back(tris[i]);
                break;
        }
    }
    if (tris.size() == leftTris.size() || tris.size() == rightTris.size()) {
        node->triangles = tris;node->leaf = true;node->box = tris[0]->getBoundingBox();
        for (long i=1; i<tris.size(); i++) {node->box.expand(tris[i]->getBoundingBox());}
        node->left = new KDNode();node->right = new KDNode();
        node->left->triangles = std::vector<Triangle*>();node->right->triangles = std::vector<Triangle*>();
        return node;
    }
    node->left = build(leftTris, depth+1);node->right = build(rightTris, depth+1);
    return node;
}

// Finds nearest triangle in kd tree that intersects with ray.
bool KDNode::hit(KDNode *node, const Ray &ray, double &t, double &tmin, glm::dvec3 &normal, glm::dvec3 &c) {
    double dist;
    if (node->box.intersection(ray, dist)){
        if (dist > tmin) return false;

        bool hitTri = false;
        bool hitLeft = false;
        bool hitRight = false;
        long triIdx;

        if (!node->leaf) {
            hitLeft = hit(node->left, ray, t, tmin, normal, c);
            hitRight = hit(node->right, ray, t, tmin, normal, c);
            return hitLeft || hitRight;
        }
        else {
            long trianglesSize = node->triangles.size();
            for (long i=0; i<trianglesSize; i++) {
                if (node->triangles[i]->intersect(ray, t, tmin, normal)){
                    hitTri = true;
                    tmin = t;
                    triIdx = i;
                }
            }
            if (hitTri) {
                glm::dvec3 p = ray.origin + ray.direction * tmin;
                c = node->triangles[triIdx]->get_colour_at(p);
                return true;
            }
        }
    }
    return false;
}