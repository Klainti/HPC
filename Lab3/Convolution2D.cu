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
  int newDim = imageW+filterR*2;

  for (y = filterR; y < imageH+filterR; y++) {
    for (x = filterR; x < imageW+filterR; x++) {
      user_data_t sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;
        sum += h_Src[y * newDim + d] * h_Filter[filterR - k];
      }
      h_Dst[y * newDim + x] = sum;
    }
  }
}

__global__ void convolutionRowGPU(user_data_t *d_Dst, user_data_t *d_Src, user_data_t *d_Filter,
                        int imageW, int imageH, int filterR) {
    int x,y,k;
    int newDim = imageW+filterR*2;
    user_data_t sum=0.0;

    x = threadIdx.x + blockDim.x * blockIdx.x + filterR;
    y = threadIdx.y + blockDim.y * blockIdx.y + filterR;

    for (k = -filterR; k <= filterR; k++) {
      int d = x + k;
      sum += d_Src[y * newDim + d] * d_Filter[filterR - k];
    }
    d_Dst[y * newDim + x] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(user_data_t *h_Dst, user_data_t *h_Src, user_data_t *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  int newDim = imageW+filterR*2;

  for (y = filterR; y < imageH+filterR; y++) {
    for (x = filterR; x < imageW+filterR; x++) {
      user_data_t sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;
        sum += h_Src[d * newDim + x] * h_Filter[filterR - k];
      }
      h_Dst[y * newDim + x] = sum;
    }
  }
}

__global__ void convolutionColumnGPU(user_data_t *d_Dst, user_data_t *d_Src, user_data_t *d_Filter,
                    int imageW, int imageH, int filterR) {
    int x, y, k;
    int newDim = imageW+filterR*2;
    user_data_t sum = 0.0;

    x = threadIdx.x + blockDim.x * blockIdx.x + filterR;
    y = threadIdx.y + blockDim.y * blockIdx.y + filterR;

    for (k = -filterR; k <= filterR; k++) {
      int d = y + k;
      sum += d_Src[d * newDim + x] * d_Filter[filterR - k];
    }
    d_Dst[y * newDim + x] = sum;
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
    int padding, image_plus_paddingW;
    int image_size;
    int a_break = 0;
    unsigned int i,j;
    user_data_t residual;

    cudaError_t error; //check if a function fails!

    //elapsed time of GPU
    GpuTimer myTimer;
    myTimer.CreateTimer();

    //elapsed time of GPU
    struct timespec  tv1, tv2;

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    // padding size
    padding = filter_radius;

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    //image width with padding
    image_plus_paddingW = imageW+2*padding;

    // new image size
    image_size = image_plus_paddingW*image_plus_paddingW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (user_data_t *)malloc(FILTER_LENGTH * sizeof(user_data_t));
    h_Input     = (user_data_t *)malloc(image_size * sizeof(user_data_t));
    h_Buffer    = (user_data_t *)malloc(image_size * sizeof(user_data_t));
    h_OutputCPU = (user_data_t *)malloc(image_size * sizeof(user_data_t));
    GPU_result = (user_data_t *)malloc(image_size * sizeof(user_data_t));

    // initialize all buffers with zeros!!
    memset(h_Input, 0.0, image_size * sizeof(user_data_t));
    memset(h_Buffer, 0.0, image_size * sizeof(user_data_t));
    memset(h_OutputCPU, 0.0, image_size * sizeof(user_data_t));
    memset(GPU_result, 0.0, image_size * sizeof(user_data_t));

    error = cudaMalloc((void**) &d_Filter, FILTER_LENGTH*sizeof(user_data_t));
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMalloc((void**) &d_Input, image_size * sizeof(user_data_t));
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMalloc((void**) &d_Buffer, image_size * sizeof(user_data_t));
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMalloc((void**) &d_OutputGPU, image_size * sizeof(user_data_t));
    if (error != cudaSuccess){
        debug_e("cudaMalloc failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // initialize all buffers with zeros!!
    error = cudaMemset(d_Buffer, 0.0, image_size*sizeof(user_data_t));
    if (error != cudaSuccess){
        debug_e("cudaMemset failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMemset(d_OutputGPU, 0.0, image_size*sizeof(user_data_t));
    if (error != cudaSuccess){
        debug_e("cudaMemset failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (user_data_t)(rand() % 16);
    }

    for (i = padding; i < imageH+padding; i++) {
        for (j = padding; j < imageW+padding; j++) {
            h_Input[i*image_plus_paddingW+j] = (user_data_t)rand() / ((user_data_t)RAND_MAX / 255) + (user_data_t)rand() / (user_data_t)RAND_MAX;
        }
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

    printf ("%10g (s)\n",(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
    			(double) (tv2.tv_sec - tv1.tv_sec));


    /* ------------------------------------------- GPU computation -----------------------------------------------------------------*/
    printf("GPU computation: ");
    //Dimensions of grib and blocks!
    dim3 grid, block;

    // max num thread per block = 32!
    // So, if imageW > 32, create max num threads per block!
    if (imageW < 32) {
        block.x = imageW;
        block.y = imageH;
        grid.x = 1;
        grid.y = 1;
    } else {
        block.x = 32;
        block.y = 32;
        grid.x = imageW/32;
        grid.y = imageH/32;
    }

    myTimer.Start(); // start the timer

    // copy host memory to device
    error = cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(user_data_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        debug_e("cudaMemcpy failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMemcpy(d_Input, h_Input, image_size * sizeof(user_data_t), cudaMemcpyHostToDevice);
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
    error = cudaMemcpy(GPU_result, d_OutputGPU, image_size*sizeof(user_data_t), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        debug_e("cudaMemcpy failed! Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    myTimer.Stop(); // stop the timer
    printf("%lf (ms)\n", myTimer.Elapsed());
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
        for (i=padding; i<imageH+padding; i++) {
            for (j=padding; j<imageW+padding; j++) {
                residual = ABS(GPU_result[i*image_plus_paddingW+j] - h_OutputCPU[i*image_plus_paddingW+j]);

                if (residual>accuracy){
                    printf("Accuracy problem! residual between CPU and GPU is: %lf\n", residual);
                    a_break = 1;
                    break;
                }
            }
            if (a_break){
                break; // go to free memory!
            }
        }
    #else //find the max residual!
        user_data_t max_residual = -1.0;
        for (i=padding; i<imageH+padding; i++) {
            for (j=padding; j<imageW+padding; j++) {
                residual = ABS(GPU_result[i*image_plus_paddingW+j] - h_OutputCPU[i*image_plus_paddingW+j]);

                if (residual>max_residual){
                    max_residual = residual;
                }
            }
        }
        printf("Max residual: %lf\n", max_residual);
    #endif

    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);
    free(GPU_result);

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
    cudaDeviceReset();

    return 0;
}
