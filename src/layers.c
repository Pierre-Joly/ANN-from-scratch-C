#include "layers.h"

DenseLayer *create_dense_layer(size_t input_size, size_t output_size,
                               size_t batch_size) {
  // Allocate memory for the DenseLayer structure
  DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));

  if (layer == NULL) {
    perror("Failed to allocate memory for DenseLayer");
    return NULL;
  }

  // Set the layer's dimensions: input size, output size, and batch size
  layer->input_size = input_size;
  layer->output_size = output_size;
  layer->batch_size = batch_size;

  // Allocate memory for weights (input_size x output_size)
  layer->weights = (float *)malloc(input_size * output_size * sizeof(float));

  if (layer->weights == NULL) {
    perror("Failed to allocate memory for weights");
    free(layer);
    return NULL;
  }

  // // Allocate memory for biases (output_size)
  layer->biases = (float *)malloc(output_size * sizeof(float));

  if (layer->biases == NULL) {
    perror("Failed to allocate memory for biases");
    free(layer->weights);
    free(layer);
    return NULL;
  }

  // Allocate memory for the output (batch_size x output_size)
  layer->output = (float *)malloc(batch_size * output_size * sizeof(float));

  if (layer->output == NULL) {
    perror("Failed to allocate memory for output");
    free(layer->weights);
    free(layer->biases);
    free(layer);
    return NULL;
  }

  // Initialize weights
  for (size_t i = 0; i < input_size * output_size; i++) {
    layer->weights[i] = 0.0f;
  }

  // Initialize biases
  for (size_t i = 0; i < output_size; i++) {
    layer->biases[i] = 0.0f;
  }

  return layer;
}

void free_dense_layer(DenseLayer *layer) {
  if (layer == NULL) {
    return;
  }
  // Free components
  free(layer->weights);
  free(layer->biases);
  free(layer->output);

  // Free layer
  free(layer);
}
