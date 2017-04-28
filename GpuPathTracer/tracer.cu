//
// Created by ashish on 4/4/17.
//


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
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cudaUtils.h"






__device__ glm::vec3 getSample(const kernelInfo & info,curandState* randstate){




    uint tx = threadIdx.x;
    uint ty = threadIdx.y;
    uint bw = blockDim.x;
    uint bh = blockDim.y;
    uint x = blockIdx.x*bw + tx;
    uint y = blockIdx.y*bh + ty;
    size_t pixelPos = y*info.width+x;
    const glm::vec4 * const triTex = info.triangleTex;
    const Sphere * const sphereTex = info.sphereTex;

    const size_t triTexSize = info.numVerts;
    const size_t sphTexSize = info.numSpheres;

    const int depth = info.depth;
    const int w = info.width;
    const int h = info.height;



//    if(x == 0 && y ==0 ) {
//        printf("rei tex size is %ld \n",info.numVerts);
//        printf("received vars\n");
//        printf("%f\n",info.cam.dist);
//        printf("%f\n",info.cam.fov);
//        printf("%f\n",info.cam.aspect);
//        printf("%d\n",info.width);
//        printf("%d\n",info.height);
//        printf("cam width %f\n",info.cam.dist*info.cam.aspect*info.cam.fov);
//        printf("tri tex size %ld\n",info.numVerts);
//    }




    u_char r = u_char(211),g = u_char(211),b = u_char(211),a = 255;


    Ray currRay = getCamRayDir(info.cam,x,y,w,h,randstate);





    {
        vec3 mask = vec3(1.0f, 1.0f, 1.0f); // colour mask
        vec3 accucolor = vec3(0.0f, 0.0f, 0.0f); // accumulated colour
        vec3 direct = vec3(0, 0, 0);

        for (unsigned int d = 0; d < depth; ++d){

            //TODO this magic num
            float tmin = 0.00001f; // set to 0.01f when using refractive material
            float tmax = 1e20;

            int minSphereIdx = -1;
            int minBoxId = -1;
            int minTriIdx = -1;
            int geomtype = GeoType::NONE;
            float scene_t = tmax;
            vec3 objcol = vec3(0, 0, 0);
            vec3 emit = vec3(0, 0, 0);
            vec3 hitpos;//pos in 3d where ray hit the closest

            vec3 n; // normal
            vec3 nl; // oriented normal
            vec3 nextdir; // ray direction of next path segment
            vec3 trinormal = vec3(0, 0, 0);
            Mat mat;

//            intersectAllTriangles(triTex,currRay,scene_t,minTriIdx,triTexSize,geomtype,info.cullBackFaces);
            //TODO enable this
//            intersectBVHandTriangles(currRay,0,F32_MAX,
//                                     info.bvhData.dev_triNode,
//                                     info.bvhData.dev_triPtr,
//                                     info.bvhData.dev_triIndicesPtr, minTriIdx, scene_t,n,geomtype,info.cullBackFaces);
            intersectAllSpeheres(sphereTex,currRay,scene_t,minSphereIdx,sphTexSize,geomtype);


//            if(scene_t < tmax){
//                scene_t = min(scene_t,45.0f);
//                scene_t = (scene_t-15)/30;
//                scene_t = sqrt(scene_t);
//                r = 255*scene_t;
//                g = 255*scene_t;
//                return vec3(scene_t,scene_t,scene_t);
//
//            }


            hitpos = currRay.origin+currRay.dir*scene_t;
            if(geomtype == GeoType::SPHERE){
                const Sphere & hS = sphereTex[minSphereIdx];
                n = hS.getNormal(hitpos);
                nl = glm::dot(n, currRay.dir) < 0 ? n : n * -1.0f;
                objcol = hS.col;;   // object colour
                emit = hS.emi;  // object emission
                mat = hS.mat;
                accucolor += (mask * emit);

            }
            else if(geomtype == GeoType::TRI){

                n = normalize(trinormal);
                nl = dot(n, currRay.dir) < 0 ? n : n * -1.0f;  // correctly oriented normal

                //TODO correct here color,mat hardcoded
                //Vec3f colour = hitTriIdx->_colorf;
                vec3 colour(0.9f, 0.3f, 0.0f); // hardcoded triangle colour  .9f, 0.3f, 0.0f
                objcol = colour;
                emit = vec3(0.0, 0.0, 0);  // object emission
                accucolor += (mask * emit);


            }
//            else if(geomtype == GeoType::BOX){
//                Box &box = boxes[box_id];
//                x = r.orig + r.dir*t;  // intersection point on object
//                n = normalize(box.normalAt(x)); // normal
//                nl = dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal
//                f = box.col;  // box colour
//                refltype = box.refl;
//                emit = box.emi; // box emission
//                accucolor += (mask * emit);
//            }



            if (mat == Mat::DIFF){

                // pick two random numbers
                float phi = 2 * M_PI * curand_uniform(randstate);
                float r2 = curand_uniform(randstate);
                float r2s = sqrtf(r2);


                vec3 nt,nb;
                nt = (fabs(nl.x) > fabs(nl.y))?vec3(nl.z, 0, -nl.x):vec3(0, -nl.z, nl.y);
                nt = normalize(nt);
                nb = cross(nl,nt);
                nb = normalize(nb);

                vec3 randVec = uniformSampleHemisphere(curand_uniform(randstate),curand_uniform(randstate));
                // compute cosine weighted random ray direction on hemisphere

                nextdir = vec3(nb.x*randVec.x+nl.x*randVec.y+nt.x*randVec.z,
                               nb.y*randVec.x+nl.y*randVec.y+nt.y*randVec.z,
                               nb.z*randVec.x+nl.z*randVec.y+nt.z*randVec.z);
                nextdir = glm::normalize(nextdir);

                // offset origin next path segment to prevent self intersection
                //TODO magic num here
                hitpos += nl * 0.001f; // scene size dependent

                // multiply mask with colour of object
                mask *= objcol;


            }


            // ideal specular reflection (mirror)
            else if (mat == Mat::SPEC){


                //Snell's law
                nextdir = currRay.dir - nl * dot(nl, currRay.dir) * 2.0f;
                nextdir = glm::normalize(nextdir);

                //TODO this magic num
                // offset origin next path segment to prevent self intersection
                hitpos += nl * 0.001f;

                // multiply mask with colour of object
                mask *= objcol;
            }
            // perfectly refractive material (glass, water)
            else if (mat == REFR){

                bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
                float nc = 1.0f;  // Index of Refraction air
                float nt = 1.4f;  // Index of Refraction glass/water
                float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
                float ddn = dot(currRay.dir, nl);
                float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

                if (cos2t < 0.0f) // total internal reflection
                {
                    nextdir = currRay.dir - n * 2.0f * dot(n, currRay.dir);
                    nextdir = normalize(nextdir);

                    //TODO here magic num
                    // offset origin next path segment to prevent self intersection
                    hitpos += nl * 0.001f; // scene size dependent
                }
                else // cos2t > 0
                {
                    // compute direction of transmission ray
                    vec3 tdir = currRay.dir * nnt;
                    tdir -= n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t)));
                    tdir = normalize(tdir);

                    float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
                    float c = 1.f - (into ? -ddn : dot(tdir, n));
                    float Re = R0 + (1.f - R0) * c * c * c * c * c;
                    float Tr = 1 - Re; // Transmission
                    float P = .25f + .5f * Re;
                    float RP = Re / P;
                    float TP = Tr / (1.f - P);

                    // randomly choose reflection or transmission ray
                    if (curand_uniform(randstate) < 0.2) // reflection ray
                    {
                        mask *= RP;
                        nextdir = currRay.dir - n * 2.0f * dot(n, currRay.dir);
                        nextdir = normalize(nextdir);

                        hitpos += nl * 0.001f; // scene size dependent
                    }
                    else // transmission ray
                    {
                        mask *= TP;
                        nextdir = tdir;
                        nextdir = normalize(nextdir);

                        hitpos += nl * 0.001f; // epsilon must be small to avoid artefacts
                    }
                }
            }
            else if(mat == METAL){

                // compute random perturbation of ideal reflection vector
                // the higher the phong exponent, the closer the perturbed vector is to the ideal reflection direction
                float phi = 2 * M_PI * curand_uniform(randstate);
                float r2 = curand_uniform(randstate);
                float phongexponent = 30;
                float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
                float sinTheta = sqrtf(1 - cosTheta * cosTheta);

                // create orthonormal basis uvw around reflection vector with hitpoint as origin
                // w is ray direction for ideal reflection
                vec3 w1 = currRay.dir - n * 2.0f * dot(n, currRay.dir); w1 = normalize(w1);
                vec3 u = cross((fabs(w1.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w1); u = normalize(u);
                vec3 v = cross(w1, u); // v is already normalised because w and u are normalised

                // compute cosine weighted random ray direction on hemisphere
                nextdir = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
                nextdir = normalize(nextdir);


                //TODO magic num here
                // offset origin next path segment to prevent self intersection
                hitpos += nl * 0.0001f;  // scene size dependent

                // multiply mask with colour of object
                mask *= objcol;



            }

            currRay.origin = hitpos;
            currRay.dir = nextdir;



        }




        return accucolor;


    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


}



__global__ void trace(const kernelInfo info){




    uint tx = threadIdx.x;
    uint ty = threadIdx.y;
    uint bw = blockDim.x;
    uint bh = blockDim.y;
    uint x = blockIdx.x*bw + tx;
    uint y = blockIdx.y*bh + ty;

    const int w = info.width;
    const int h = info.height;

    if(x>=w || y>=h)return;
    size_t pixelPos = y*info.width+x;


    curandState randState; // state of the random number generator, to prevent repetition
    curand_init(info.hash + (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x, 0, 0, &randState);



    //TODO add background color here

    uchar3 c3 = make_uchar3(255,255,255);
    vec3 finalcol(0,0,0);



    finalcol += getSample(info,&randState);





    if(x == 0 && y == 0){
        printf("cam dirt%d\n",info.cam.dirty);
    }



    vec3 initVal = info.accumBuffer[pixelPos];
    initVal*=(info.constantPdf-1);
    initVal+=finalcol;
    initVal*=(1.0f/info.constantPdf);
    initVal = glm::clamp(initVal,vec3(0.0f,0.0f,0.0f),vec3(1.0f,1.0f,1.0f));
    info.accumBuffer[pixelPos] = initVal;


    c3.x*=info.accumBuffer[pixelPos].x;
    c3.y*=info.accumBuffer[pixelPos].y;
    c3.z*=info.accumBuffer[pixelPos].z;

    info.dev_drawRes[pixelPos] = rgbToUint(c3.x,c3.y,c3.z);

}



#include <iostream>
void BasicScene::launchKernel(const kernelInfo &info) {

//    using namespace std;
//    cout << width <<" " << height << endl;




     dim3 blocks((info.width+info.blockSize.x)/info.blockSize.x,(info.height+info.blockSize.y)/info.blockSize.y,1);
    trace<<<blocks,info.blockSize>>>(info);
}


