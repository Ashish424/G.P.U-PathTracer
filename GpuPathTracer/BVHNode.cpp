//
// Created by tgamerz on 21/4/17.
//

#include "BVHNode.hpp"
#include <vector>


int BVHNode::getSubtreeSize(BVH_STAT stat) const  // recursively counts some type of nodes (either leafnodes, innernodes, childnodes) or unmber of triangles
{
    int cnt;

    switch (stat)
    {
        default: assert(0);  // unknown mode
        case BVH_STAT_NODE_COUNT:      cnt = 1; break; // counts all nodes including leafnodes
        case BVH_STAT_LEAF_COUNT:      cnt = isLeaf() ? 1 : 0; break; // counts only leafnodes
        case BVH_STAT_INNER_COUNT:     cnt = isLeaf() ? 0 : 1; break; // counts only innernodes
        case BVH_STAT_TRIANGLE_COUNT:  cnt = isLeaf() ? reinterpret_cast<const LeafNode*>(this)->getNumTriangles() : 0; break; // counts all triangles
        case BVH_STAT_CHILDNODE_COUNT: cnt = getNumChildNodes(); break; ///counts only childnodes
    }

    if (!isLeaf()) // if current node is not a leaf node, continue counting its childnodes recursively
    {
        for (int i = 0; i<getNumChildNodes(); i++)
            cnt += getChildNode(i)->getSubtreeSize(stat);
    }

    return cnt;
}


void BVHNode::deleteSubtree()
{
    for (int i = 0; i<getNumChildNodes(); i++)
        getChildNode(i)->deleteSubtree();

    delete this;
}


void BVHNode::computeSubtreeProbabilities(const Platform& p, float probability, float& sah)
{
    sah += probability * p.getCost(this->getNumChildNodes(), this->getNumTriangles());

    m_probability = probability;

    // recursively compute probabilities and add to SAH
    for (int i = 0; i<getNumChildNodes(); i++)
    {
        BVHNode* child = getChildNode(i);
        child->m_parentProbability = probability;           /// childnode area / parentnode area
        child->computeSubtreeProbabilities(p, probability * child->m_bounds.area() / this->m_bounds.area(), sah);
    }
}


// TODO: requires valid probabilities...
float BVHNode::computeSubtreeSAHCost(const Platform& p) const
{
    float SAH = m_probability * p.getCost(getNumChildNodes(), getNumTriangles());

    for (int i = 0; i<getNumChildNodes(); i++)
        SAH += getChildNode(i)->computeSubtreeSAHCost(p);

    return SAH;
}

//-------------------------------------------------------------

void assignIndicesDepthFirstRecursive(BVHNode* node, int& index, bool includeLeafNodes)
{
    if (node->isLeaf() && !includeLeafNodes)
        return;

    node->m_index = index++;
    for (int i = 0; i<node->getNumChildNodes(); i++)
        assignIndicesDepthFirstRecursive(node->getChildNode(i), index, includeLeafNodes);
}

void BVHNode::assignIndicesDepthFirst(int index, bool includeLeafNodes)
{
    assignIndicesDepthFirstRecursive(this, index, includeLeafNodes);
}

//-------------------------------------------------------------

void BVHNode::assignIndicesBreadthFirst(int index, bool includeLeafNodes)
{
    std::vector<BVHNode*> nodes;
    nodes.push_back(this);
    int head = 0;

    while (head < nodes.size())
    {
        // pop
        BVHNode* node = nodes[head++];

        // discard
        if (node->isLeaf() && !includeLeafNodes)
            continue;

        // assign
        node->m_index = index++;

        // push children
        for (int i = 0; i<node->getNumChildNodes(); i++)
            nodes.push_back(node->getChildNode(i));
    }
}