#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    File:         Makefile                                                  */
#    Description:  Makefile for 2Dconvolution								 */
#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

NVCC = nvcc
OPTFLAGS = -g -fmad=false -O4
DEBUGFLAFS = -Xcompiler -DDEBUG -Xcompiler -DDEBUG_COLOR
MEMFLAGS = -Xcompiler -rdynamic -lineinfo

H_FILES = debug.h

Convolution2D: Convolution2D.cu
	$(NVCC) $(OPTFLAGS) $(MEMFLAGS)  $(DEBUGFLAFS) -o Convolution2D Convolution2D.cu

clean:
	rm -rf *.o
