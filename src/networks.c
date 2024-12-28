#include "networks.h"

#include "errors.h"

DenseNetwork *create_dense_network(
    int num_layers, int batch_size,
    void (*activation)(const float *restrict, float *restrict, const int,
                       const int),
    void (*output_activation)(const float *restrict, float *restrict, const int,
                              const int)) {
  // Allocate memory for the DenseNetwork structure
  DenseNetwork *network = (DenseNetwork *)malloc(sizeof(DenseNetwork));

  if (network == NULL) {
    perror("Failed to allocate memory for DenseNetwork");
    return NULL;
  }

  // Set static components
  if (num_layers == 0) {
    fprintf(stderr, "Error: Cannot define a Network with no layers\n");
    free(network);
    return NULL;
  }
  network->num_layers = num_layers;
  if (batch_size == 0) {
    fprintf(stderr, "Error: Cannot define a Network with 0 batch size\n");
    free(network);
    return NULL;
  }
  network->batch_size = batch_size;
  network->activation = activation;
  network->output_activation = output_activation;

  // Allocate memory for the array of layer pointers
  network->layers =
      (DenseLayer **)malloc((size_t)num_layers * sizeof(DenseLayer *));

  if (network->layers == NULL) {
    perror("Failed to allocate memory for DenseLayer pointers");
    free(network);
    return NULL;
  }

  // Initialize all layer pointers to NULL
  for (int i = 0; i < num_layers; i++) {
    network->layers[i] = NULL;
  }

  return network;
}

int add_layer(DenseNetwork *network, int index, DenseLayer *layer) {
  if (index >= network->num_layers) {
    fprintf(stderr, "Error: Index %d is out of bounds for network layers\n",
            index);
    return ERR_OUT_OF_BOUNDS;
  }

  if (index != 0 && network->layers[index - 1] == NULL) {
    fprintf(stderr,
            "Error: The preceding layer at index %d has not been initialized\n",
            index - 1);
    return ERR_LAYER_NOT_INITIALIZED;
  }

  if (network->layers[index] != NULL) {
    fprintf(stderr,
            "Error: The layer at index %d has already been initialized\n",
            index);
    return ERR_LAYER_ALREADY_SET;
  }

  if (network->batch_size != layer->batch_size) {
    fprintf(stderr,
            "Error: The batch size of layer (%d) does not match the network's "
            "batch size (%d)\n",
            layer->batch_size, network->batch_size);
    return ERR_BATCH_SIZE_MISMATCH;
  }

  if (index != 0 &&
      network->layers[index - 1]->output_size != layer->input_size) {
    fprintf(stderr,
            "Error: Dimension mismatch - preceding layer output size (%d) "
            "does not match new layer input size (%d)\n",
            network->layers[index - 1]->output_size, layer->input_size);
    return ERR_DIMENSION_MISMATCH;
  }

  network->layers[index] = layer;

  return SUCCESS;  // Operation succeeded
}

void free_dense_network(DenseNetwork *network) {
  if (network == NULL) {
    return;  // Nothing to free
  }

  // Free each layer in the network
  for (int i = 0; i < network->num_layers; i++) {
    if (network->layers[i] != NULL) {
      free_dense_layer(network->layers[i]);
    }
  }

  // Free the array of DenseLayer pointers
  free(network->layers);
  network->layers = NULL;

  // Free the DenseNetwork structure
  free(network);
}
