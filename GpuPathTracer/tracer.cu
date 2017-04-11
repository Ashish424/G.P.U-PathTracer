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
//#include "cutil_math.h"  // required for float3 vector math
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "curand.h"
#include "curand_kernel.h"

#include "cudaUtils.cu"



//passing struct by value
__global__ void cudaProcess(const kernelInfo info){


    uint tx = threadIdx.x;
    uint ty = threadIdx.y;
    uint bw = blockDim.x;
    uint bh = blockDim.y;
    uint x = blockIdx.x*bw + tx;
    uint y = blockIdx.y*bh + ty;
    size_t pixelPos = y*info.width+x;
    int w = info.width;
    int h = info.height;
    if(x == 0 && y ==0 ) {
        //TODO keep this function
        printf("received vars\n");
        printf("%f\n",info.cam.dist);
        printf("%f\n",info.cam.fov);
        printf("%f\n",info.cam.aspect);

        printf("%d\n",info.width);
        printf("%d\n",info.height);

    }
    if(x>=w || y>=h)
        return;



    u_char r = 255,g = 255,b = 255,a = 255;

    Ray camRay = getCamRayDir(info.cam,x,h-1-y,w,h);




    //sphere test
    {

        Sphere sp(20*4.99985f,vec3(0.0f, 0,-100),vec3(0,0,0),vec3(0.9f, 0.9f, 0.9f ), DIFF);
        float dist = sp.intersect(camRay);
//        if(dist >0 ){
//            printf("%d %d\n",x,y);
//        }
        if(dist > 0 ){
            r = 255;
            g = 0;
            b = 0;
            a = 255;
        }
    }

    uchar4 c4 = make_uchar4(r, g, b, a);
    info.dev_drawRes[(y)*info.width+x] = rgbToInt(c4.x,c4.y,c4.z);

}

#include <iostream>
void BasicScene::launchKernel(const kernelInfo &info) {

    using namespace std;
//    cout << width <<" " << height << endl;


     dim3 blocks((info.width+info.blockSize.x)/info.blockSize.x,(info.height+info.blockSize.y)/info.blockSize.y,1);
     cudaProcess<<<blocks,info.blockSize>>>(info);
}