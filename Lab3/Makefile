#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    File:         Makefile                                                  */
#    Description:  Makefile for 2Dconvolution								 */
#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

NVCC = nvcc
OPTFLAGS = -g -O4

# For debugging purpose, check debug.h for more details!
DEBUGFLAFS = -Xcompiler -DDEBUG -Xcompiler -DDEBUG_COLOR

# Comment this if you want to use FLOATS. Default double precision!
DOUBLE_OR_FLOAT = -Xcompiler -DUSE_DOUBLES

# Checks the accuracy of the convolution between CPU and GPU.
# Comment this if you want to calculate the max residual between them.
CHECK_ACCURACY = -Xcompiler -DCHECK_ACCURACY

Convolution2D: Convolution2D.cu
	$(NVCC) $(OPTFLAGS) $(DOUBLE_OR_FLOAT) $(CHECK_ACCURACY) $(DEBUGFLAFS) -o Convolution2D Convolution2D.cu

clean:
	rm -rf *.o
