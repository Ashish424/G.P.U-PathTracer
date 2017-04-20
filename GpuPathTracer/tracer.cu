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
    const size_t triTexSize = info.numTris;
    int w = info.width;
    int h = info.height;
    if(x == 0 && y ==0 ) {

//        printf("rei tex size is %ld \n",info.numTris);
        //TODO keep this function and disable it
//        printf("received vars\n");
//        printf("%f\n",info.cam.dist);
//        printf("%f\n",info.cam.fov);
//        printf("%f\n",info.cam.aspect);
//        printf("%d\n",info.width);
//        printf("%d\n",info.height);
//        printf("cam width %f\n",info.cam.dist*info.cam.aspect*info.cam.fov);
//        printf("tri tex size %ld\n",info.numTris);
    }

    if(x>=w || y>=h)
        return;


    //test texture cuda

//    float4 col  = tex1Dfetch<float4>(info.triangleTex,pixelPos);
//    glm::vec4 newVec = f4tov(col);
//    col = vtof4(newVec);


    u_char r = 255,g = 255,b = 255,a = 255;

    Ray camRay = getCamRayDir(info.cam,x,y,w,h);

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
        intersectAllTriangles(triTex,camRay, t, triangle_id, triTexSize, geomtype);






        if(t<inf){
            r = 255;
            g = 0;
            b = 0;
            a = 255;
        }


//        }

//         t is distance to closest intersection of ray with all primitives in the scene (spheres, boxes and triangles)
//        return t<inf;

    }



    //sphere test
//    {
//
//        float rad= 300/(sqrt(2.0f)-1);
//        Sphere sp(rad/2,make_float3(0.0f, 0,-rad-h/2),make_float3(0,0,0),make_float3(0.9f, 0.9f, 0.9f ), DIFF);
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
