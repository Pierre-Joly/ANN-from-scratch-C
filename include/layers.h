#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int input_size;
  int output_size;
  int batch_size;
  float *weights;
  float *biases;
  float *output;
} DenseLayer;

DenseLayer *create_dense_layer(int input_size, int output_size, int batch_size);
void free_dense_layer(DenseLayer *layer);

#endif  // LAYERS_H
