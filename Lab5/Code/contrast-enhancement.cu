#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

#define CDF_SIZE 256

__global__ void prescan(int *g_odata, int *g_idata, int n){

    __shared__ int temp[CDF_SIZE];
    __shared__ int doubled_tmp[CDF_SIZE];

    int thid = threadIdx.x;
    int offset = 1;
    
    //load into shared mem
    temp[2*thid] = g_idata[2*thid];
    doubled_tmp[2*thid] = g_idata[2*thid];

    temp[2*thid + 1] = g_idata[2*thid + 1];
    doubled_tmp[2*thid + 1] = g_idata[2*thid + 1];

    for (int d = n >> 1; d > 0; d >>=1 ) {// build sum in place up the tree

        __syncthreads();

        if (thid < d) {
            int ai = offset*(2*thid+1) - 1;
            int bi = offset*(2*thid+2) - 1;
            
            //printf("thread: %d adds %d and %d\n", thid, ai, bi);
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
 
    if (thid == 0) {
        temp[n-1] = temp[n-2];
    }

    for (int d = 1; d < n; d *= 2){ // traverse down tree and build scan
        offset >>= 1;
        __syncthreads(); // wait for up-sweep phase to finish

        if (thid < d) {
            int ai = offset*(2*thid+1) - 1;
            int bi = offset*(2*thid+2) - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;

        }
    }

    __syncthreads(); // wait for down-sweep phase to finish
    // write result to output
    g_odata[2*thid] = temp[2*thid] + doubled_tmp[2*thid];
    g_odata[2*thid + 1] = temp[2*thid + 1] + doubled_tmp[2*thid + 1];
    
}

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    int *h_cdf;
    
    // variables for device!
    int *d_cdf, *d_hist;
    cudaError_t error;
    
    //dimensions of kernel
    dim3 grid, block;
    block.x = 128; //half of CDF_SIZE!
    grid.x = 1;

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        
    h_cdf = (int*) malloc(CDF_SIZE*sizeof(int));

    // allocate mem for d_cdf
    error = cudaMalloc((void **)&d_cdf, CDF_SIZE*sizeof(int));
    if (error != cudaSuccess){
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // allocate mem for d_cdf
    error = cudaMalloc((void **)&d_hist, CDF_SIZE*sizeof(int));
    if (error != cudaSuccess){
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

    // copy hist to device input
    error = cudaMemcpy(d_hist, hist, CDF_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      printf("Error during cudaMemcpy of d_lut to h_lut:  %s\n", cudaGetErrorString(error));
      exit(-1);
    }
 
    prescan <<< grid, block >>> (d_cdf,d_hist, 256);

    // copy hist to device input
    error = cudaMemcpy(h_cdf, d_cdf, CDF_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      printf("Error during cudaMemcpy of d_lut to h_lut:  %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    
    for (int i=0; i<CDF_SIZE; i++){
        printf("%d: %d\n", i,h_cdf[i]);
    }

    return result;
}
