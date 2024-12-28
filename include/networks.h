#ifndef NETWORKS_H
#define NETWORKS_H

#include "layers.h"

typedef struct {
  int num_layers;
  int batch_size;
  DenseLayer **layers;
  void (*activation)(const float *restrict, float *restrict, const int,
                     const int);
  void (*output_activation)(const float *restrict, float *restrict, const int,
                            const int);
} DenseNetwork;

DenseNetwork *create_dense_network(
    int num_layers, int batch_size,
    void (*activation)(const float *restrict, float *restrict, const int,
                       const int),
    void (*output_activation)(const float *restrict, float *restrict, const int,
                              const int));
int add_layer(DenseNetwork *network, int index, DenseLayer *layer);
void free_dense_network(DenseNetwork *network);

#endif  // NETWORKS_H
