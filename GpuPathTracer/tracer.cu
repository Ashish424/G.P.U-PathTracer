//
// Created by ashish on 4/4/17.
//

#include "device_launch_parameters.h"
#include "BasicScene.hpp"
#include "cuda_runtime.h"
#include <stdio.h>

__global__ void cudaProcess(unsigned int * writeRes,int width,int height){


    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;
    if(x>=width || y>=height)
        return;
    writeRes[y*width+x] = 1;
    writeRes[y*width+x]<<=30;
    writeRes[y*width+x]-=1;









}
void BasicScene::launchKernel(const BasicScene::kernelInfo &info) {



     dim3 blocks(16,16,1);
     cudaProcess<<<blocks,info.blockSize>>>(info.drawRes,info.width,info.height);
}