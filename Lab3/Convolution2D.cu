/*
* This sample implements a separable convolution
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

#include "debug.h"

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005

#define NUM_BLOCKS 1

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter,
                       int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}

__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter,
                        int imageW, int imageH, int filterR) {
    int x,y,k;
    float sum=0.0;

    x = threadIdx.x + blockDim.x * blockIdx.x;
    y = threadIdx.y + blockDim.y * blockIdx.y;

    for (k = -filterR; k <= filterR; k++) {
      int d = x + k;

      if (d >= 0 && d < imageW) {
        sum += d_Src[y * imageW + d] * d_Filter[filterR - k];
      }

      d_Dst[y * imageW + x] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}

__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter,
                    int imageW, int imageH, int filterR) {
    int x, y, k;
    float sum = 0.0;

    x = threadIdx.x + blockDim.x * blockIdx.x;
    y = threadIdx.y + blockDim.y * blockIdx.y;

    for (k = -filterR; k <= filterR; k++) {
      int d = y + k;

      if (d >= 0 && d < imageH) {
        sum += d_Src[d * imageW + x] * d_Filter[filterR - k];
      }

      d_Dst[y * imageW + x] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU;

    float
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU;

    float
    *GPU_result;

    int imageW;
    int imageH;
    unsigned int i;
    float residual;
    cudaError_t error;

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    GPU_result = (float *)malloc(imageW * imageH * sizeof(float));

    error = cudaMalloc((void**) &d_Filter, FILTER_LENGTH*sizeof(float));
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed!");
        exit(-1);
    }

    error = cudaMalloc((void**) &d_Input, imageW * imageH * sizeof(float));
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMalloc((void**) &d_Buffer, imageW * imageH * sizeof(float));
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMalloc((void**) &d_OutputGPU, imageW * imageH * sizeof(float));
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }

    // copy host memory to device
    error = cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        debug_e("cudaMemcpy failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    //kernel invocation
    dim3 grid, block;
    block.x = imageW/NUM_BLOCKS;
    block.y = imageH/NUM_BLOCKS;
    grid.x = NUM_BLOCKS;
    grid.y = NUM_BLOCKS;

    convolutionRowGPU<<<grid, block>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius); // convolution kata grammes

    // wait for convolutionRowGPU to finish!
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        debug_e("cudaSynchronize failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    convolutionColumnGPU<<<grid, block>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    //Done with computation, return result to CPU
    error = cudaMemcpy(GPU_result, d_OutputGPU, imageW*imageH*sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        debug_e("cudaMemcpy failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // ask CUDA for the last error to occur (if one exists)
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // something's gone wrong
      // print out the CUDA error as a string
      debug_e("CUDA Error: %s\n", cudaGetErrorString(error));

      // we can't recover from the error -- exit the program
      exit(-1);
    }
    // no error occurred, proceed as usual


    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas
    for (i=0; i<imageW*imageH; i++) {
        residual = ABS(GPU_result[i] - h_OutputCPU[i]);

        if (residual>accuracy){
            debug_w("Accuracy: %lf", residual);
            exit(-1);
        }
    }


    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    error = cudaFree(d_Filter);
    if (error != cudaSuccess){
        debug_e("cudaFree failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaFree(d_Input);
    if (error != cudaSuccess){
        debug_e("cudaFree failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaFree(d_Buffer);
    if (error != cudaSuccess){
        debug_e("cudaFree failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaFree(d_OutputGPU);
    if (error != cudaSuccess){
        debug_e("cudaFree failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    // cudaDeviceReset();


    return 0;
}
