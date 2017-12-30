#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;

    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}
