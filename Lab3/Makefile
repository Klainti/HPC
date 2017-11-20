#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    File:         Makefile                                                  */
#    Description:  Makefile for 2Dconvolution								 */
#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

NVCC = nvcc
OPTFLAGS = -g -G -O4
DEBUGFLAFS = -DDEBUG -DDEBUG_COLOR

H_FILES = debug.h

Convolution2D: Convolution2D.cu
	$(NVCC) $(OPTFLAGS) $(DEBUGFLAFS) -o Convolution2D Convolution2D.cu

clean:
	rm -rf *.o
