#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values

    //  to take care of padding wrt even & odd size kernels
    int start = (l.size - 1) /2;

    int b, ci, wi, hi, i, j;
    int c = l.channels;
    
    // looping over batches of in (input)
    for (b = 0; b < in.rows; b++){
        for (ci = 0; ci < c; ci++){
            for (hi = 0; hi < outh; hi++){
                for (wi = 0; wi < outw; wi++){
                    //c*b*outh*outw + outh*outw*ci + hi*outw + wi
                    int index = ((ci + c*b)*outh + hi)*outw + wi;
                    // float max = 0.0;
                    float max = -FLT_MAX;
                    
                    for (i = 0; i < l.size; i++){
                        for (j = 0; j < l.size; j++){
                            int y_coord = hi*l.stride + i - start;
                            int x_coord = wi*l.stride + j - start;
                            
                            int curr_index = ((ci + c*b)*l.height + y_coord)*l.width + x_coord;

                            // check if the x,y coord are within layer input bounds or not
                            int valid_coord = (x_coord >= 0 && y_coord >= 0  && x_coord < l.width && y_coord < l.height);

                            float val = (valid_coord != 0) ? in.data[curr_index] : -FLT_MAX;
                            max = (val > max) ? val : max;
                        }
                    }
                    out.data[index] = max;
                }
            }
        }
    }

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

    //  to take care of padding wrt even & odd size kernels
    int start = (l.size - 1) /2;

    int b, ci, wi, hi, i, j;
    int c = l.channels;

    // looping over batches of in (input)
    for (b = 0; b < in.rows; b++) {
        for (ci = 0; ci < c; ci++){
            for (hi = 0; hi < outh; hi++){
                for (wi = 0; wi < outw; wi++){
                    //c*b*outh*outw + outh*outw*ci + hi*outw + wi
                    int index = ((ci + c*b)*outh + hi)*outw + wi;
                    float max_val = delta.data[index];
                    //float max = 0.0;
                    float max = -FLT_MAX;
                    int max_index = -1;

                    for (i = 0; i < l.size; i++){
                        for (j = 0; j < l.size; j++){
                            int y_coord = hi*l.stride + i - start;
                            int x_coord = wi*l.stride + j - start;

                            int curr_index = ((ci+c*b)*l.height + y_coord)*l.width + x_coord;
                            
                            // check if the x,y coord are within layer input bounds or not
                            int valid_coord = (x_coord >= 0 && y_coord >= 0 && x_coord < l.width && y_coord < l.height);
                            float val = (valid_coord != 0) ? in.data[curr_index] : -FLT_MAX;
                           
                            // max = (val > max) ? val : max;
                            // max_index = (val >= max) ? curr_index : max_index;
                            max_index = (val > max) ? curr_index : max_index;
                            max = (val > max) ? val : max;
                        }
                    }
                    if (prev_delta.data) {
                        prev_delta.data[max_index] += max_val;    
                    }
                    
                }
            }
        }
    }
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}