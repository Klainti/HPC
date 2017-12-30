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
__constant__ int lut_const[CDF_SIZE];

__global__ void prescan(int *lut, int n, int min, int diff){

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
 
    if (thid == 0) { // clear the last element
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

    // make exclusive to inclusive and calc the lut table
    int res_ai = (int)(((float)(temp[ai + bankOffsetA] + inclusive_number1) - min)*255/diff + 0.5);    
    int res_bi = (int)(((float)(temp[bi + bankOffsetB] + inclusive_number2) - min)*255/diff + 0.5);

    // results between [0...255]
    if (res_ai < 0){
        res_ai = 0;
    } 
    else if (res_ai > 255){
        res_ai = 255;
    } 

    if (res_bi < 0){
        res_bi = 0;
    } 
    else if (res_bi > 255){
        res_bi = 255;
    }

    lut[ai] = res_ai;
    lut[bi] = res_bi;
}

// apply histogram equalization from GPU
__global__ void hist_equalGPU(unsigned char * img_out, int img_size){

    int thid = threadIdx.x + blockDim.x*blockIdx.x;

    if (thid < img_size){
        img_out[thid] =  (unsigned char)lut_const[img_out[thid]];
    }
}

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int *hist;
    int *h_lut;
    int i=0, min=0, img_size;
    double total_time;

    // variables for device!
    int *d_lut;
    unsigned char *d_result;
    cudaError_t error;
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    //dimensions of kernel
    dim3 grid, block;

    //elapsed time of GPU
    GpuTimer myTimer;
    myTimer.CreateTimer();

    //elapsed time of CPU
    struct timespec  tv1, tv2;

    // CPU MALLOC
    result.w = img_in.w;
    result.h = img_in.h;
    img_size = result.w * result.h;

    ////////////////////////// MEM ALLOCATION /////////////////////////////
    
    // need pinned memory for stream!
    error = cudaMallocHost((void **)&result.img, img_size*sizeof(unsigned char));
    if (error != cudaSuccess){
        printf("cudaMallocHost failed: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // need pinned memory for stream!
    error = cudaMallocHost((void **)&hist, CDF_SIZE*sizeof(int));
    if (error != cudaSuccess){
        printf("cudaMallocHost failed: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    h_lut = (int*) malloc(CDF_SIZE*sizeof(int));
    if (h_lut == NULL){
        printf("Malloc failed");
        exit(-1);
    }

    // allocate mem for d_lut    
    error = cudaMalloc((void **)&d_lut, CDF_SIZE*sizeof(int));
    if (error != cudaSuccess){
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // allocate mem for d_result
    error = cudaMalloc((void **)&d_result, img_size * sizeof(unsigned char));
    if (error != cudaSuccess){
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    ////////////////////////// CALC OF HIST /////////////////////////////

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    histogram(hist, img_in.img, img_in.h * img_in.w, 256);

    while(min == 0){
        min = hist[i++];
    }
 
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    total_time = (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
    			(double) (tv2.tv_sec - tv1.tv_sec);
       
    ////////////////////// CALC OF LUT ////////////////////////////////////////

    myTimer.Start(); // start the timer

    //copy hist to constant mem
    error = cudaMemcpyToSymbolAsync(d_hist, hist, 256 * sizeof(int), 0, cudaMemcpyHostToDevice,stream1);
    if (error != cudaSuccess) {
      printf("cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    block.x = CDF_SIZE/2;  
    grid.x = 1;

    // kernel invocation!
    prescan <<< grid, block, 0, stream1 >>> (d_lut, 256, min,result.w*result.h-min);

    /////////////////// APPLY HIST EQUALIZATION ////////////////////

    // Apply hist-equalization on image (kernel)
    block.x = 1024; //use max threads per block
    grid.x = int(img_size/1024 + 1);

    // copy lut to constant mem of the GPU
    error = cudaMemcpyToSymbolAsync(lut_const, d_lut, 256 * sizeof(int),0 , cudaMemcpyDeviceToDevice, stream1);
    if (error != cudaSuccess) {
        printf("cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // copy img_in to d_result to apply hist-equalization
    error = cudaMemcpyAsync(d_result, img_in.img, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice, stream1);
    if (error != cudaSuccess) {
      printf("cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    // kernel invocation!
    hist_equalGPU <<< grid, block, 0, stream1 >>> (d_result, img_size);

    // copy results from kernel to result!
    error = cudaMemcpyAsync(result.img, d_result, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream1);
    if (error != cudaSuccess) {
      printf("cudaMemcpy failed: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    cudaStreamDestroy(stream1);
    myTimer.Stop();
    total_time += (double) (myTimer.Elapsed()/1000);
    myTimer.DestroyTimer();
    // DONE!

    printf("%lf\n", total_time);
    
    // free memory
    free(h_lut);

    error = cudaFreeHost(hist);
    if (error != cudaSuccess) {
        printf("cudaFree failed: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaFree(d_lut);
    if (error != cudaSuccess) {
        printf("cudaFree failed: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaFree(d_result);
    if (error != cudaSuccess) {
        printf("cudaFree failed: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return result;
}
