#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include "gputimer.h"
#include <time.h>

#define CDF_SIZE 256

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5 

#define CONFLICT_FREE_OFFSET(n)\
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

__constant__ int d_hist[CDF_SIZE];

__global__ void prescan(int *g_odata, int n){

    __shared__ int temp[CDF_SIZE];

    int thid = threadIdx.x;
    int offset = 1;
    int inclusive_number1,inclusive_number2;
    int ai = thid;
    int bi = thid + (n/2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(ai);
    
    //load into shared mem
    inclusive_number1 = d_hist[ai];
    temp[ai + bankOffsetA] = inclusive_number1;

    inclusive_number2 = d_hist[bi];
    temp[bi + bankOffsetB] = inclusive_number2;

    for (int d = n >> 1; d > 0; d >>=1 ) {// build sum in place up the tree

        __syncthreads();

        if (thid < d) {
            int ai = offset*(2*thid+1) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);

            int bi = offset*(2*thid+2) - 1;
            bi += CONFLICT_FREE_OFFSET(bi);
                
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
 
    if (thid == 0) {
        temp[n-1 + CONFLICT_FREE_OFFSET(n-1)] = 0;
    }

    for (int d = 1; d < n; d *= 2){ // traverse down tree and build scan
        offset >>= 1;
        __syncthreads();

        if (thid < d) {
            int ai = offset*(2*thid+1) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);

            int bi = offset*(2*thid+2) - 1;
            bi += CONFLICT_FREE_OFFSET(bi);
 
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;

        }
    }

    __syncthreads(); // wait for down-sweep phase to finish

    // write result to output and convert exclusive to inclusive!!
    g_odata[ai] = temp[ai + bankOffsetA] + inclusive_number1;
    g_odata[bi] = temp[bi + bankOffsetB] + inclusive_number2;
}

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    int *h_cdf;
    
    // variables for device!
    int *d_cdf;
    cudaError_t error;
    
    //dimensions of kernel
    dim3 grid, block;
    block.x = 128; //half of CDF_SIZE!
    grid.x = 1;

    //elapsed time of GPU
    GpuTimer myTimer;
    myTimer.CreateTimer();

    //elapsed time of GPU
    struct timespec  tv1, tv2;

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        
    h_cdf = (int*) malloc(CDF_SIZE*sizeof(int));
    if (h_cdf == NULL){
        printf("Malloc failed");
        exit(-1);
    }

    // allocate mem for d_cdf
    error = cudaMalloc((void **)&d_cdf, CDF_SIZE*sizeof(int));
    if (error != cudaSuccess){
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    histogram(hist, img_in.img, img_in.h * img_in.w, 256);

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    printf ("histogram: %10g (s)\n",(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
    			(double) (tv2.tv_sec - tv1.tv_sec));


    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    printf ("histogram_equal: %10g (s)\n",(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
    			(double) (tv2.tv_sec - tv1.tv_sec));


    error = cudaMemcpyToSymbol(d_hist, hist, 256 * sizeof(int));
    if (error != cudaSuccess) {
      printf("cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    myTimer.Start(); // start the timer

    prescan <<< grid, block >>> (d_cdf, 256);

    cudaDeviceSynchronize();
    myTimer.Stop();
    printf("cuda prefix: %lf (s)\n", myTimer.Elapsed()/1000);
    myTimer.DestroyTimer();

    // copy device output to h_cdf
    error = cudaMemcpy(h_cdf, d_cdf, CDF_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      printf("cudaMemcpy failed: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    /*
    for (int i=0; i<CDF_SIZE; i++){
        printf("%d: %d\n", i,h_cdf[i]);
    }
    */
    free(h_cdf);

    error = cudaFree(d_cdf);
    if (error != cudaSuccess) {
        printf("cudaFree failed: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
 
    return result;
}
