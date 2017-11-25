/*
* This sample implements a separable convolution
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "debug.h"
#include "gputimer.h"

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005

// use doubles or float ?
#ifdef USE_DOUBLES
typedef double user_data_t;
#else
typedef float user_data_t;
#endif

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(user_data_t *h_Dst, user_data_t *h_Src, user_data_t *h_Filter,
                       int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      user_data_t sum = 0;

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

__global__ void convolutionRowGPU(user_data_t *d_Dst, user_data_t *d_Src, user_data_t *d_Filter,
                        int imageW, int imageH, int filterR) {
    int x,y,k;
    user_data_t sum=0.0;

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
void convolutionColumnCPU(user_data_t *h_Dst, user_data_t *h_Src, user_data_t *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      user_data_t sum = 0;

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

__global__ void convolutionColumnGPU(user_data_t *d_Dst, user_data_t *d_Src, user_data_t *d_Filter,
                    int imageW, int imageH, int filterR) {
    int x, y, k;
    user_data_t sum = 0.0;

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

    user_data_t
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU;

    user_data_t
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU;

    user_data_t
    *GPU_result;

    int imageW;
    int imageH;
    int num_blocks;
    unsigned int i;
    user_data_t residual;

    cudaError_t error; //check if a function fails!

    //elapsed time of GPU
    GpuTimer myTimer;
    myTimer.CreateTimer();

    //elapsed time of GPU
    struct timespec  tv1, tv2;

    debug_e("Entet number of blocks : ");
    scanf("%d", &num_blocks);

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
    h_Filter    = (user_data_t *)malloc(FILTER_LENGTH * sizeof(user_data_t));
    h_Input     = (user_data_t *)malloc(imageW * imageH * sizeof(user_data_t));
    h_Buffer    = (user_data_t *)malloc(imageW * imageH * sizeof(user_data_t));
    h_OutputCPU = (user_data_t *)malloc(imageW * imageH * sizeof(user_data_t));
    GPU_result = (user_data_t *)malloc(imageW * imageH * sizeof(user_data_t));

    error = cudaMalloc((void**) &d_Filter, FILTER_LENGTH*sizeof(user_data_t));
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed!");
        exit(-1);
    }

    error = cudaMalloc((void**) &d_Input, imageW * imageH * sizeof(user_data_t));
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMalloc((void**) &d_Buffer, imageW * imageH * sizeof(user_data_t));
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMalloc((void**) &d_OutputGPU, imageW * imageH * sizeof(user_data_t));
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (user_data_t)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (user_data_t)rand() / ((user_data_t)RAND_MAX / 255) + (user_data_t)rand() / (user_data_t)RAND_MAX;
    }

    /* ------------------------------------------- CPU computation -----------------------------------------------------------------*/

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation: ");

    /* This is the main computation. Get the starting time. */
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    /* This is the end of the main computation. */
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

    printf ("%10g\n",(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
    			(double) (tv2.tv_sec - tv1.tv_sec));


    /* ------------------------------------------- GPU computation -----------------------------------------------------------------*/
    printf("GPU computation: ");
    //Dimensions of grib and blocks!
    dim3 grid, block;
    block.x = imageW/num_blocks;
    block.y = imageH/num_blocks;
    grid.x = num_blocks;
    grid.y = num_blocks;

    myTimer.Start(); // start the timer

    // copy host memory to device
    error = cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(user_data_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        debug_e("cudaMemcpy failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(user_data_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // kernel for row convolution
    convolutionRowGPU<<<grid, block>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius); // convolution kata grammes

    // wait for convolutionRowGPU to finish!
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        debug_e("cudaSynchronize failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // kernel for column convolution!
    convolutionColumnGPU<<<grid, block>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    //Done with computation, return result to CPU
    error = cudaMemcpy(GPU_result, d_OutputGPU, imageW*imageH*sizeof(user_data_t), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        debug_e("cudaMemcpy failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    myTimer.Stop(); // stop the timer
    printf("%lf\n", myTimer.Elapsed());
    myTimer.DestroyTimer();

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
    #ifdef CHECK_ACCURACY
        for (i=0; i<imageW*imageH; i++) {
            residual = ABS(GPU_result[i] - h_OutputCPU[i]);

            if (residual>accuracy){
                debug_w("Accuracy: %lf", residual);
                exit(-1);
            }
        }
    #else //find the max residual!
        user_data_t max_residual = -1.0;
        for (i=0; i<imageW*imageH; i++) {
            residual = ABS(GPU_result[i] - h_OutputCPU[i]);

            if (residual>max_residual){
                max_residual = residual;
            }
        }
        printf("Max residual: %lf\n", max_residual);
    #endif

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
    //cudaDeviceReset();

    return 0;
}
