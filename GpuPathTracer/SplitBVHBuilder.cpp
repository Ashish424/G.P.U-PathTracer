//
// Created by tgamerz on 21/4/17.
//


#include "SplitBVHBuilder.hpp"
#include "Sort.hpp"

using glm::vec3;
using glm::ivec3;
//helper function
inline float getVecFloat(const glm::vec3 &a, int dim) { if(dim == 0)return a.x; else if(dim == 1)return a.y; else if(dim == 2)return a.z; else assert(false); }
inline int getVecInt(const glm::ivec3 &a, int dim) { if(dim == 0)return a.x; else if(dim == 1)return a.y; else if(dim == 2)return a.z; else assert(false);  }
inline void setVecFloat(glm::vec3 &a, int dim, float val){ if(dim == 0) a.x = val; else if(dim == 1) a.y = val; else if(dim == 2)a.z = val; else assert(false); }
SplitBVHBuilder::SplitBVHBuilder(BVH& bvh, const BVH::BuildParams& params)
        : m_bvh(bvh),
          m_platform(bvh.getPlatform()),
          m_params(params),
          m_minOverlap(0.0f),   /// overlap of AABBs
          m_sortDim(-1)
{
}

//------------------------------------------------------------------------

SplitBVHBuilder::~SplitBVHBuilder(void)
{
}

//------------------------------------------------------------------------

BVHNode* SplitBVHBuilder::run(void)  /// returns the rootnode
{

    // See SBVH paper by Martin Stich for details

    // Initialize reference stack and determine root bounds.

    const SceneMesh::Triangle* tris = m_bvh.getSceneMesh()->getTrianglePtr(); // list of all triangles in scene
    const vec3* verts = m_bvh.getSceneMesh()->getVertexPtr();  // list of all vertices in scene

    NodeSpec rootSpec;
    rootSpec.numRef = m_bvh.getSceneMesh()->getNumTriangles();  // number of triangles/references in entire scene (root)
    m_refStack.resize(rootSpec.numRef);

    // calculate the bounds of the rootnode by merging the AABBs of all the references
    for (int i = 0; i < rootSpec.numRef; i++)
    {
        // assign triangle to the array of references
        m_refStack[i].triIdx = i;

        // grow the bounds of each reference AABB in all 3 dimensions by including the vertex
        m_refStack[i].bounds.grow(verts[tris[i].vertices.x]);
        m_refStack[i].bounds.grow(verts[tris[i].vertices.y]);
        m_refStack[i].bounds.grow(verts[tris[i].vertices.z]);

        rootSpec.bounds.grow(m_refStack[i].bounds);
    }

    // Initialize rest of the members.

    m_minOverlap = rootSpec.bounds.area() * m_params.splitAlpha;  /// split alpha (maximum allowable overlap) relative to size of rootnode
    m_rightBounds.resize(std::max(rootSpec.numRef, (int)NumSpatialBins) - 1);
    m_numDuplicates = 0;
    //m_progressTimer.start();

    // Build recursively.
    BVHNode* root = buildNode(rootSpec, 0, 0.0f, 1.0f);  /// actual building of splitBVH
    m_bvh.getTriIndices().clear();   // removes unused memoryspace from triIndices array

    // Done.

    if (m_params.enablePrints)
        printf("SplitBVHBuilder: progress %.0f%%, duplicates %.0f%%\n",
               100.0f, (float)m_numDuplicates / (float)m_bvh.getSceneMesh()->getNumTriangles() * 100.0f);

    return root;
}

//------------------------------------------------------------------------

int SplitBVHBuilder::sortCompare(void* data, int idxA, int idxB)
{
    const SplitBVHBuilder* ptr = (const SplitBVHBuilder*)data;
    int dim = ptr->m_sortDim;
    const Reference& ra = ptr->m_refStack[idxA];  // ra is a reference (struct containing a triIdx and bounds)
    const Reference& rb = ptr->m_refStack[idxB];  //
    float ca = getVecFloat(ra.bounds.min(), dim) + getVecFloat(ra.bounds.max(), dim);
    float cb = getVecFloat(rb.bounds.min(), dim) + getVecFloat(rb.bounds.max(), dim);
    return (ca < cb) ? -1 : (ca > cb) ? 1 : (ra.triIdx < rb.triIdx) ? -1 : (ra.triIdx > rb.triIdx) ? 1 : 0;
}

//------------------------------------------------------------------------

void SplitBVHBuilder::sortSwap(void* data, int idxA, int idxB)
{
    SplitBVHBuilder* ptr = (SplitBVHBuilder*)data;
    swap(ptr->m_refStack[idxA], ptr->m_refStack[idxB]);
}

//------------------------------------------------------------------------

inline float min1f3(const float& a, const float& b, const float& c){ return std::min(std::min(a, b), c); }

BVHNode* SplitBVHBuilder::buildNode(const NodeSpec& spec, int level, float progressStart, float progressEnd)
{
    // Display progress.

//	if (m_params.enablePrints && m_progressTimer.getElapsed() >= 1.0f)
//	{
//		printf("SplitBVHBuilder: progress %.0f%%, duplicates %.0f%%\r",
//			progressStart * 100.0f, (float)m_numDuplicates / (float)m_bvh.getScene()->getNumTriangles() * 100.0f);
//		m_progressTimer.start();
//	}

    // Small enough or too deep => create leaf.

    if (spec.numRef <= m_platform.getMinLeafSize() || level >= MaxDepth)
    {
        return createLeaf(spec);
    }

    // Find split candidates.

    float area = spec.bounds.area();
    float leafSAH = area * m_platform.getTriangleCost(spec.numRef);
    float nodeSAH = area * m_platform.getNodeCost(2);
    ObjectSplit object = findObjectSplit(spec, nodeSAH);

    SpatialSplit spatial;
    if (level < MaxSpatialDepth)
    {
        AABB overlap = object.leftBounds;
        overlap.intersect(object.rightBounds);
        if (overlap.area() >= m_minOverlap)
            spatial = findSpatialSplit(spec, nodeSAH);
    }

    // Leaf SAH is the lowest => create leaf.

    float minSAH = min1f3(leafSAH, object.sah, spatial.sah);
    if (minSAH == leafSAH && spec.numRef <= m_platform.getMaxLeafSize()){
        return createLeaf(spec);
    }

    // Leaf SAH is not the lowest => Perform spatial split.

    NodeSpec left, right;
    if (minSAH == spatial.sah){
        performSpatialSplit(left, right, spec, spatial);
    }

    if (!left.numRef || !right.numRef){ /// if either child contains no triangles/references
        performObjectSplit(left, right, spec, object);
    }

    // Create inner node.

    m_numDuplicates += left.numRef + right.numRef - spec.numRef;
    float progressMid = lerp(progressStart, progressEnd, (float)right.numRef / (float)(left.numRef + right.numRef));
    BVHNode* rightNode = buildNode(right, level + 1, progressStart, progressMid);
    BVHNode* leftNode = buildNode(left, level + 1, progressMid, progressEnd);
    return new InnerNode(spec.bounds, leftNode, rightNode);
}

//------------------------------------------------------------------------

BVHNode* SplitBVHBuilder::createLeaf(const NodeSpec& spec)
{
    std::vector<int>& tris = m_bvh.getTriIndices();

    for (int i = 0; i < spec.numRef; i++){
        tris.push_back(m_refStack.back().triIdx); // take a triangle from the stack and add it to tris array
        m_refStack.pop_back();
    }
    return new LeafNode(spec.bounds, tris.size() - spec.numRef, tris.size());
}

//------------------------------------------------------------------------

SplitBVHBuilder::ObjectSplit SplitBVHBuilder::findObjectSplit(const NodeSpec& spec, float nodeSAH)
{
    ObjectSplit split;
    const Reference* refPtr = &m_refStack[m_refStack.size() - spec.numRef];

    // Sort along each dimension.

    for (m_sortDim = 0; m_sortDim < 3; m_sortDim++)
    {
        Sort(m_refStack.size() - spec.numRef, m_refStack.size(), this, sortCompare, sortSwap);

        // Sweep right to left and determine bounds.

        AABB rightBounds;
        for (int i = spec.numRef - 1; i > 0; i--)
        {
            rightBounds.grow(refPtr[i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.

        AABB leftBounds;
        for (int i = 1; i < spec.numRef; i++)
        {
            leftBounds.grow(refPtr[i - 1].bounds);
            float sah = nodeSAH + leftBounds.area() * m_platform.getTriangleCost(i) + m_rightBounds[i - 1].area() * m_platform.getTriangleCost(spec.numRef - i);
            if (sah < split.sah)
            {
                split.sah = sah;
                split.sortDim = m_sortDim;
                split.numLeft = i;
                split.leftBounds = leftBounds;
                split.rightBounds = m_rightBounds[i - 1];
            }
        }
    }
    return split;
}

//------------------------------------------------------------------------

void SplitBVHBuilder::performObjectSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split)
{
    m_sortDim = split.sortDim;
    Sort(m_refStack.size() - spec.numRef, m_refStack.size(), this, sortCompare, sortSwap);

    left.numRef = split.numLeft;
    left.bounds = split.leftBounds;
    right.numRef = spec.numRef - split.numLeft;
    right.bounds = split.rightBounds;
}

//------------------------------------------------------------------------

// clamping functions for int, float and Vec3i
inline int clamp1i(const int v, const int lo, const int hi){ return v < lo ? lo : v > hi ? hi : v; }
inline float clamp1f(const float v, const float lo, const float hi){ return v < lo ? lo : v > hi ? hi : v; }

inline ivec3 clamp3i(const ivec3& v, const ivec3& lo, const ivec3& hi){
    return ivec3(clamp1i(v.x, lo.x, hi.x), clamp1i(v.y, lo.y, hi.y), clamp1i(v.z, lo.z, hi.z));}


SplitBVHBuilder::SpatialSplit SplitBVHBuilder::findSpatialSplit(const NodeSpec& spec, float nodeSAH)
{
    // Initialize bins.

    vec3 origin = spec.bounds.min();
    vec3 binSize = (spec.bounds.max() - origin) * (1.0f / (float)NumSpatialBins);
    vec3 invBinSize = vec3(1.0f / binSize.x, 1.0f / binSize.y, 1.0f / binSize.z);

    for (int dim = 0; dim < 3; dim++)
    {
        for (int i = 0; i < NumSpatialBins; i++)
        {
            SpatialBin& bin = m_bins[dim][i];
            bin.bounds = AABB();
            bin.enter = 0;
            bin.exit = 0;
        }
    }

    // Chop references into bins.

    for (int refIdx = m_refStack.size() - spec.numRef; refIdx < m_refStack.size(); refIdx++)
    {
        const Reference& ref = m_refStack[refIdx];

        ivec3 firstBin = clamp3i(ivec3((ref.bounds.min() - origin) * invBinSize), ivec3(0, 0, 0), ivec3(NumSpatialBins - 1, NumSpatialBins - 1, NumSpatialBins - 1));
        ivec3 lastBin = clamp3i(ivec3((ref.bounds.max() - origin) * invBinSize), firstBin, ivec3(NumSpatialBins - 1, NumSpatialBins - 1, NumSpatialBins - 1));

        for (int dim = 0; dim < 3; dim++)
        {
            Reference currRef = ref;
            for (int i = getVecInt(firstBin, dim); i < getVecInt(lastBin, dim); i++)
            {
                Reference leftRef, rightRef;
                splitReference(leftRef, rightRef, currRef, dim, getVecFloat(origin, dim) + getVecFloat(binSize, dim) * (float)(i + 1));
                m_bins[dim][i].bounds.grow(leftRef.bounds);
                currRef = rightRef;
            }
            m_bins[dim][getVecInt(lastBin, dim)].bounds.grow(currRef.bounds);
            m_bins[dim][getVecInt(firstBin, dim)].enter++;
            m_bins[dim][getVecInt(lastBin, dim)].exit++;
        }
    }

    // Select best split plane.

    SpatialSplit split;
    for (int dim = 0; dim < 3; dim++)
    {
        // Sweep right to left and determine bounds.

        AABB rightBounds;
        for (int i = NumSpatialBins - 1; i > 0; i--)
        {
            rightBounds.grow(m_bins[dim][i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.

        AABB leftBounds;
        int leftNum = 0;
        int rightNum = spec.numRef;

        for (int i = 1; i < NumSpatialBins; i++)
        {
            leftBounds.grow(m_bins[dim][i - 1].bounds);
            leftNum += m_bins[dim][i - 1].enter;
            rightNum -= m_bins[dim][i - 1].exit;

            float sah = nodeSAH + leftBounds.area() * m_platform.getTriangleCost(leftNum) + m_rightBounds[i - 1].area() * m_platform.getTriangleCost(rightNum);
            if (sah < split.sah)
            {
                split.sah = sah;
                split.dim = dim;
                split.pos = getVecFloat(origin, dim) + getVecFloat(binSize, dim) * (float)i;
            }
        }
    }
    return split;
}

//------------------------------------------------------------------------

void SplitBVHBuilder::performSpatialSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SpatialSplit& split)
{
    // Categorize references and compute bounds.
    //
    // Left-hand side:      [leftStart, leftEnd[
    // Uncategorized/split: [leftEnd, rightStart[
    // Right-hand side:     [rightStart, refs.size()[

    std::vector<Reference>& refs = m_refStack;
    int leftStart = refs.size() - spec.numRef;
    int leftEnd = leftStart;
    int rightStart = refs.size();
    left.bounds = right.bounds = AABB();

    for (int i = leftEnd; i < rightStart; i++)
    {
        // Entirely on the left-hand side?

        if (getVecFloat(refs[i].bounds.max(), split.dim) <= split.pos)
        {
            left.bounds.grow(refs[i].bounds);
            swap(refs[i], refs[leftEnd++]);
        }

            // Entirely on the right-hand side?

        else if (getVecFloat(refs[i].bounds.min(), split.dim) >= split.pos)
        {
            right.bounds.grow(refs[i].bounds);
            swap(refs[i--], refs[--rightStart]);
        }
    }

    // Duplicate or unsplit references intersecting both sides.

    while (leftEnd < rightStart)
    {
        // Split reference.

        Reference lref, rref;
        splitReference(lref, rref, refs[leftEnd], split.dim, split.pos);

        // Compute SAH for duplicate/unsplit candidates.

        AABB lub = left.bounds;  // Unsplit to left:     new left-hand bounds.
        AABB rub = right.bounds; // Unsplit to right:    new right-hand bounds.
        AABB ldb = left.bounds;  // Duplicate:           new left-hand bounds.
        AABB rdb = right.bounds; // Duplicate:           new right-hand bounds.
        lub.grow(refs[leftEnd].bounds);
        rub.grow(refs[leftEnd].bounds);
        ldb.grow(lref.bounds);
        rdb.grow(rref.bounds);

        float lac = m_platform.getTriangleCost(leftEnd - leftStart);
        float rac = m_platform.getTriangleCost(refs.size() - rightStart);
        float lbc = m_platform.getTriangleCost(leftEnd - leftStart + 1);
        float rbc = m_platform.getTriangleCost(refs.size() - rightStart + 1);

        float unsplitLeftSAH = lub.area() * lbc + right.bounds.area() * rac;
        float unsplitRightSAH = left.bounds.area() * lac + rub.area() * rbc;
        float duplicateSAH = ldb.area() * lbc + rdb.area() * rbc;
        float minSAH = min1f3(unsplitLeftSAH, unsplitRightSAH, duplicateSAH);

        // Unsplit to left?

        if (minSAH == unsplitLeftSAH)
        {
            left.bounds = lub;
            leftEnd++;
        }

            // Unsplit to right?

        else if (minSAH == unsplitRightSAH)
        {
            right.bounds = rub;
            swap(refs[leftEnd], refs[--rightStart]);
        }

            // Duplicate?

        else
        {
            left.bounds = ldb;
            right.bounds = rdb;
            refs[leftEnd++] = lref;
            refs.push_back(rref);
        }
    }

    left.numRef = leftEnd - leftStart;
    right.numRef = refs.size() - rightStart;
}

//------------------------------------------------------------------------

void SplitBVHBuilder::splitReference(Reference& left, Reference& right, const Reference& ref, int dim, float pos)
{
    // Initialize references.

    left.triIdx = right.triIdx = ref.triIdx;
    left.bounds = right.bounds = AABB();

    // Loop over vertices/edges.
    const ivec3& inds = m_bvh.getSceneMesh()->getTriangle(ref.triIdx).vertices;
    const vec3* verts = m_bvh.getSceneMesh()->getVertexPtr();
    const vec3* v1 = &verts[inds.z];

    for (int i = 0; i < 3; i++)
    {
        const vec3* v0 = v1;
        v1 = &verts[getVecInt(inds, i)];
        float v0p = getVecFloat((*v0), dim);
        float v1p = getVecFloat((*v1), dim);

        // Insert vertex to the boxes it belongs to.

        if (v0p <= pos)
            left.bounds.grow(*v0);
        if (v0p >= pos)
            right.bounds.grow(*v0);

        // Edge intersects the plane => insert intersection to both boxes.

        if ((v0p < pos && v1p > pos) || (v0p > pos && v1p < pos))
        {
            vec3 t = lerp(*v0, *v1, clamp1f((pos - v0p) / (v1p - v0p), 0.0f, 1.0f));
            left.bounds.grow(t);
            right.bounds.grow(t);
        }
    }

    // Intersect with original bounds.


    setVecFloat(left.bounds.max(),dim, pos);
    setVecFloat(right.bounds.min(), dim, pos);
    left.bounds.intersect(ref.bounds);
    right.bounds.intersect(ref.bounds);
}

//------------------------------------------------------------------------
