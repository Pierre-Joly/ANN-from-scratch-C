#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
  size_t input_size;
  size_t output_size;
  size_t batch_size;
  float *weights;
  float *biases;
  float *output;
} DenseLayer;

DenseLayer *create_dense_layer(size_t input_size, size_t output_size,
                               size_t batch_size);
void free_dense_layer(DenseLayer *layer);

#endif  // LAYERS_H
