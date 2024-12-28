#include "networks.h"

#include "errors.h"

DenseNetwork *create_dense_network(size_t num_layers, size_t batch_size,
                                   float (*activation)(float),
                                   float (*output_activation)(float)) {
  // Allocate memory for the DenseNetwork structure
  DenseNetwork *network = (DenseNetwork *)malloc(sizeof(DenseNetwork));

  if (network == NULL) {
    perror("Failed to allocate memory for DenseNetwork");
    return NULL;
  }

  // Set static components
  network->num_layers = num_layers;
  network->batch_size = batch_size;
  network->activation = activation;
  network->output_activation = output_activation;

  // Allocate memory for the array of layer pointers
  network->layers = (DenseLayer **)malloc(num_layers * sizeof(DenseLayer *));

  if (network->layers == NULL) {
    perror("Failed to allocate memory for DenseLayer pointers");
    free(network);
    return NULL;
  }

  // Initialize all layer pointers to NULL
  for (size_t i = 0; i < num_layers; i++) {
    network->layers[i] = NULL;
  }

  return network;
}

int add_layer(DenseNetwork *network, size_t index, DenseLayer *layer) {
  if (index >= network->num_layers) {
    fprintf(stderr, "Error: Index %zu is out of bounds for network layers\n",
            index);
    return ERR_OUT_OF_BOUNDS;
  }

  if (index != 0 && network->layers[index - 1] == NULL) {
    fprintf(
        stderr,
        "Error: The preceding layer at index %zu has not been initialized\n",
        index - 1);
    return ERR_LAYER_NOT_INITIALIZED;
  }

  if (network->layers[index] != NULL) {
    fprintf(stderr,
            "Error: The layer at index %zu has already been initialized\n",
            index);
    return ERR_LAYER_ALREADY_SET;
  }

  if (network->batch_size != layer->batch_size) {
    fprintf(stderr,
            "Error: The batch size of layer (%zu) does not match the network's "
            "batch size (%zu)\n",
            layer->batch_size, network->batch_size);
    return ERR_BATCH_SIZE_MISMATCH;
  }

  if (index != 0 &&
      network->layers[index - 1]->output_size != layer->input_size) {
    fprintf(stderr,
            "Error: Dimension mismatch - preceding layer output size (%zu) "
            "does not match new layer input size (%zu)\n",
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
  for (size_t i = 0; i < network->num_layers; i++) {
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
