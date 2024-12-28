#ifndef NETWORKS_H
#define NETWORKS_H

#include "layers.h"

typedef struct {
  size_t num_layers;
  size_t batch_size;
  DenseLayer **layers;
  float (*activation)(float);
  float (*output_activation)(float);
} DenseNetwork;

DenseNetwork *create_dense_network(size_t num_layers, size_t batch_size,
                                   float (*activation)(float),
                                   float (*output_activation)(float));
int add_layer(DenseNetwork *network, size_t index, DenseLayer *layer);
void free_dense_network(DenseNetwork *network);

#endif  // NETWORKS_H
