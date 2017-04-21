//
// Created by ashish on 4/4/17.
//

//TODO remove this header
#include "device_launch_parameters.h"


#include "BasicScene.hpp"
#include "cuda_runtime.h"
#include <stdio.h>
#include "math.h"
#include <math_functions.h>
#include <vector_functions.h>
#include <vector_types.h>
#include "vector_functions.h"
#include "device_launch_parameters.h"
#include "cutil_math.h"  // required for float3 vector math
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cudaUtils.cu"






__device__ int counter = 0;
__global__ void cudaProcess(const kernelInfo info){




    uint tx = threadIdx.x;
    uint ty = threadIdx.y;
    uint bw = blockDim.x;
    uint bh = blockDim.y;
    uint x = blockIdx.x*bw + tx;
    uint y = blockIdx.y*bh + ty;
    size_t pixelPos = y*info.width+x;
    const glm::vec4 * const triTex = info.triangleTex;
    const int depth = info.depth;
    const size_t triTexSize = info.numVerts;
    const int w = info.width;
    const int h = info.height;
    if(x == 0 && y ==0 ) {

//        printf("rei tex size is %ld \n",info.numVerts);
        //TODO keep this function and disable it
//        printf("received vars\n");
//        printf("%f\n",info.cam.dist);
//        printf("%f\n",info.cam.fov);
//        printf("%f\n",info.cam.aspect);
//        printf("%d\n",info.width);
//        printf("%d\n",info.height);
//        printf("cam width %f\n",info.cam.dist*info.cam.aspect*info.cam.fov);
//        printf("tri tex size %ld\n",info.numVerts);
    }

    if(x>=w || y>=h)
        return;


    //test texture cuda

//    float4 col  = tex1Dfetch<float4>(info.triangleTex,pixelPos);
//    glm::vec4 newVec = f4tov(col);
//    col = vtof4(newVec);


    u_char r = 255,g = 255,b = 255,a = 255;

    Ray camRay = getCamRayDir(info.cam,x,y,w,h);
    
    
    
//    {
//        vec3 mask = vec3(1.0f, 1.0f, 1.0f); // colour mask
//        vec3 accucolor = vec3(0.0f, 0.0f, 0.0f); // accumulated colour
//        vec3 direct = vec3(0, 0, 0);
//
//        for (unsigned int d = 0; d < depth; ++d){
//
//            int hitSphereIdx = -1;
//            int hitTriIdx = -1;
//            int bestTriIdx = -1;
//            int geomtype = -1;
//            float hitSphereDist = 1e20;
//            float hitDistance = 1e20;
//            float scene_t = 1e20;
//            vec3 objcol = vec3(0, 0, 0);
//            vec3 emit = vec3(0, 0, 0);
//            vec3 hitpoint; // intersection point
//            vec3 n; // normal
//            vec3 nl; // oriented normal
//            vec3 nextdir; // ray direction of next path segment
//            vec3 trinormal = vec3(0, 0, 0);
//            Refl refltype;
//            float ray_tmin = 0.00001f; // set to 0.01f when using refractive material
//            float ray_tmax = 1e20;
//
//            // intersect all triangles in the scene stored in BVH
//
//            int debugbingo = 0;
//            intersectBVHandTriangles(make_float4(rayorig.x, rayorig.y, rayorig.z, ray_tmin), make_float4(raydir.x, raydir.y, raydir.z, ray_tmax),
//                                     gpuNodes, gpuTriWoops, gpuDebugTris, gpuTriIndices, bestTriIdx, hitDistance, debugbingo, trinormal, leafcount, tricount, false);
//
//            //DEBUGintersectBVHandTriangles(make_float4(rayorig.x, rayorig.y, rayorig.z, ray_tmin), make_float4(raydir.x, raydir.y, raydir.z, ray_tmax),
//            //gpuNodes, gpuTriWoops, gpuDebugTris, gpuTriIndices, bestTriIdx, hitDistance, debugbingo, trinormal, leafcount, tricount, false);
//
//
//            // intersect all spheres in the scene
//
//            // float3 required for sphere intersection (to avoid "dynamic allocation not allowed" error)
//            float3 rayorig_flt3 = make_float3(rayorig.x, rayorig.y, rayorig.z);
//            float3 raydir_flt3 = make_float3(raydir.x, raydir.y, raydir.z);
//
//            float numspheres = sizeof(spheres) / sizeof(Sphere);
//            for (int i = int(numspheres); i--;)  // for all spheres in scene
//                // keep track of distance from origin to closest intersection point
//                if ((hitSphereDist = spheres[i].intersect(Ray(rayorig_flt3, raydir_flt3))) && hitSphereDist < scene_t && hitSphereDist > 0.01f){
//                    scene_t = hitSphereDist; hitSphereIdx = i; geomtype = 1; }
//
//            if (hitDistance < scene_t && hitDistance > ray_tmin) // triangle hit
//            {
//                scene_t = hitDistance;
//                hitTriIdx = bestTriIdx;
//                geomtype = 2;
//            }
//
//            // sky gradient colour
//            //float t = 0.5f * (raydir.y + 1.2f);
//            //vec3 skycolor = vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + vec3(0.9f, 0.3f, 0.0f) * t;
//
//
//            // SPHERES:
//            if (geomtype == 1){
//                Sphere &hitsphere = spheres[hitSphereIdx]; // hit object with closest intersection
//                hitpoint = rayorig + raydir * scene_t;  // intersection point on object
//                n = vec3(hitpoint.x - hitsphere.pos.x, hitpoint.y - hitsphere.pos.y, hitpoint.z - hitsphere.pos.z);	// normal
//                n.normalize();
//                nl = dot(n, raydir) < 0 ? n : n * -1; // correctly oriented normal
//                objcol = vec3(hitsphere.col.x, hitsphere.col.y, hitsphere.col.z);   // object colour
//                emit = vec3(hitsphere.emi.x, hitsphere.emi.y, hitsphere.emi.z);  // object emission
//                refltype = hitsphere.refl;
//                accucolor += (mask * emit);
//            }
//
//            // TRIANGLES:
//            if (geomtype == 2){
//
//                //pBestTri = &pTriangles[triangle_id];
//                hitpoint = rayorig + raydir * scene_t; // intersection point
//
//                // float4 normal = tex1Dfetch(triNormalsTexture, pBestTriIdx);
//                n = trinormal;
//                n.normalize();
//                nl = dot(n, raydir) < 0 ? n : n * -1;  // correctly oriented normal
//                //vec3 colour = hitTriIdx->_colorf;
//                vec3 colour = vec3(0.9f, 0.3f, 0.0f); // hardcoded triangle colour  .9f, 0.3f, 0.0f
//                refltype = COAT; // objectmaterial
//                objcol = colour;
//                emit = vec3(0.0, 0.0, 0);  // object emission
//                accucolor += (mask * emit);
//            }
//
//            // basic material system, all parameters are hard-coded (such as phong exponent, index of refraction)
//
//            // diffuse material, based on smallpt by Kevin Beason
//            if (refltype == DIFF){
//
//                // pick two random numbers
//                float phi = 2 * M_PI * curand_uniform(randstate);
//                float r2 = curand_uniform(randstate);
//                float r2s = sqrtf(r2);
//
//                // compute orthonormal coordinate frame uvw with hitpoint as origin
//                vec3 w = nl; w.normalize();
//                vec3 u = cross((fabs(w.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w); u.normalize();
//                vec3 v = cross(w, u);
//
//                // compute cosine weighted random ray direction on hemisphere
//                nextdir = u*cosf(phi)*r2s + v*sinf(phi)*r2s + w*sqrtf(1 - r2);
//                nextdir.normalize();
//
//                // offset origin next path segment to prevent self intersection
//                hitpoint += nl * 0.001f; // scene size dependent
//
//                // multiply mask with colour of object
//                mask *= objcol;
//
//            } // end diffuse material
//
//            // Phong metal material from "Realistic Ray Tracing", P. Shirley
//            if (refltype == METAL){
//
//                // compute random perturbation of ideal reflection vector
//                // the higher the phong exponent, the closer the perturbed vector is to the ideal reflection direction
//                float phi = 2 * M_PI * curand_uniform(randstate);
//                float r2 = curand_uniform(randstate);
//                float phongexponent = 30;
//                float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
//                float sinTheta = sqrtf(1 - cosTheta * cosTheta);
//
//                // create orthonormal basis uvw around reflection vector with hitpoint as origin
//                // w is ray direction for ideal reflection
//                vec3 w = raydir - n * 2.0f * dot(n, raydir); w.normalize();
//                vec3 u = cross((fabs(w.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w); u.normalize();
//                vec3 v = cross(w, u); // v is already normalised because w and u are normalised
//
//                // compute cosine weighted random ray direction on hemisphere
//                nextdir = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
//                nextdir.normalize();
//
//                // offset origin next path segment to prevent self intersection
//                hitpoint += nl * 0.0001f;  // scene size dependent
//
//                // multiply mask with colour of object
//                mask *= objcol;
//            }
//
//            // ideal specular reflection (mirror)
//            if (refltype == SPEC){
//
//                // compute relfected ray direction according to Snell's law
//                nextdir = raydir - n * dot(n, raydir) * 2.0f;
//                nextdir.normalize();
//
//                // offset origin next path segment to prevent self intersection
//                hitpoint += nl * 0.001f;
//
//                // multiply mask with colour of object
//                mask *= objcol;
//            }
//
//
//            // COAT material based on https://github.com/peterkutz/GPUPathTracer
//            // randomly select diffuse or specular reflection
//            // looks okay-ish but inaccurate (no Fresnel calculation yet)
//            if (refltype == COAT){
//
//                float rouletteRandomFloat = curand_uniform(randstate);
//                float threshold = 0.05f;
//                vec3 specularColor = vec3(1, 1, 1);  // hard-coded
//                bool reflectFromSurface = (rouletteRandomFloat < threshold); //computeFresnel(make_vec3(n.x, n.y, n.z), incident, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient);
//
//                if (reflectFromSurface) { // calculate perfectly specular reflection
//
//                    // Ray reflected from the surface. Trace a ray in the reflection direction.
//                    // TODO: Use Russian roulette instead of simple multipliers!
//                    // (Selecting between diffuse sample and no sample (absorption) in this case.)
//
//                    mask *= specularColor;
//                    nextdir = raydir - n * 2.0f * dot(n, raydir);
//                    nextdir.normalize();
//
//                    // offset origin next path segment to prevent self intersection
//                    hitpoint += nl * 0.001f; // scene size dependent
//                }
//
//                else {  // calculate perfectly diffuse reflection
//
//                    float r1 = 2 * M_PI * curand_uniform(randstate);
//                    float r2 = curand_uniform(randstate);
//                    float r2s = sqrtf(r2);
//
//                    // compute orthonormal coordinate frame uvw with hitpoint as origin
//                    vec3 w = nl; w.normalize();
//                    vec3 u = cross((fabs(w.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w); u.normalize();
//                    vec3 v = cross(w, u);
//
//                    // compute cosine weighted random ray direction on hemisphere
//                    nextdir = u*cosf(r1)*r2s + v*sinf(r1)*r2s + w*sqrtf(1 - r2);
//                    nextdir.normalize();
//
//                    // offset origin next path segment to prevent self intersection
//                    hitpoint += nl * 0.001f;  // // scene size dependent
//
//                    // multiply mask with colour of object
//                    mask *= objcol;
//                }
//            } // end COAT
//
//            // perfectly refractive material (glass, water)
//            // set ray_tmin to 0.01 when using refractive material
//            if (refltype == REFR){
//
//                bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
//                float nc = 1.0f;  // Index of Refraction air
//                float nt = 1.4f;  // Index of Refraction glass/water
//                float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
//                float ddn = dot(raydir, nl);
//                float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);
//
//                if (cos2t < 0.0f) // total internal reflection
//                {
//                    nextdir = raydir - n * 2.0f * dot(n, raydir);
//                    nextdir.normalize();
//
//                    // offset origin next path segment to prevent self intersection
//                    hitpoint += nl * 0.001f; // scene size dependent
//                }
//                else // cos2t > 0
//                {
//                    // compute direction of transmission ray
//                    vec3 tdir = raydir * nnt;
//                    tdir -= n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t)));
//                    tdir.normalize();
//
//                    float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
//                    float c = 1.f - (into ? -ddn : dot(tdir, n));
//                    float Re = R0 + (1.f - R0) * c * c * c * c * c;
//                    float Tr = 1 - Re; // Transmission
//                    float P = .25f + .5f * Re;
//                    float RP = Re / P;
//                    float TP = Tr / (1.f - P);
//
//                    // randomly choose reflection or transmission ray
//                    if (curand_uniform(randstate) < 0.2) // reflection ray
//                    {
//                        mask *= RP;
//                        nextdir = raydir - n * 2.0f * dot(n, raydir);
//                        nextdir.normalize();
//
//                        hitpoint += nl * 0.001f; // scene size dependent
//                    }
//                    else // transmission ray
//                    {
//                        mask *= TP;
//                        nextdir = tdir;
//                        nextdir.normalize();
//
//                        hitpoint += nl * 0.001f; // epsilon must be small to avoid artefacts
//                    }
//                }
//            }
//
//            // set up origin and direction of next path segment
//            rayorig = hitpoint;
//            raydir = nextdir;
//        } // end bounces for loop
//
//
//    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    {
        float t;
        int triangle_id;
        int geomtype = -1;

        float tmin = 1e20;
        float tmax = -1e20;

        float d = 1e21;
        float k = 1e21;
        float q = 1e21;
        float inf = t = 1e20;





        // if ray hits bounding box of triangle meshes, intersect ray with all triangles
        //TODO insert bounding box here
        intersectAllTriangles(triTex,camRay, t, triangle_id, triTexSize, geomtype,info.cullBackFaces);






        if(t<inf){
            t = min(45.0f,t);
            t-=15;
            t/=30;
            r = 255*t;
            g = 0;
            b = 0;
            a = 255;
        }



//         t is distance to closest intersection of ray with all primitives in the scene (spheres, boxes and triangles)
//        return t<inf;

    }



    //sphere test
//    {
//
//        float rad= 300/(sqrt(2.0f)-1);
//        Sphere sp(rad/2,vec3(0.0f, 0,-rad-h/2),vec3(0,0,0),vec3(0.9f, 0.9f, 0.9f ), DIFF);
//        float dist = sp.intersect(camRay);
//
//        if(dist > 0 ){
//            r = 0;
//            g = 255;
//            b = 0;
//            a = 255;
//        }
//    }


    uchar4 c4 = make_uchar4(r, g, b, a);
//    uchar4 c4
    info.dev_drawRes[pixelPos] = rgbToUint(c4.x,c4.y,c4.z);

}

#include <iostream>
void BasicScene::launchKernel(const kernelInfo &info) {

//    using namespace std;
//    cout << width <<" " << height << endl;


     dim3 blocks((info.width+info.blockSize.x)/info.blockSize.x,(info.height+info.blockSize.y)/info.blockSize.y,1);
     cudaProcess<<<blocks,info.blockSize>>>(info);
}
