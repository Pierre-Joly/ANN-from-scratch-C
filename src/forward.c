#include "forward.h"

#include "operations.h"

static inline void process_dense_layer(
    const DenseLayer *layer, const float *input, float *output, int batch_size,
    void (*activation)(const float *, float *, const int, const int)) {
  int input_size = layer->input_size;
  int output_size = layer->output_size;

  // Perform matrix multiplication
  matmul(layer->weights, input, output, output_size, batch_size, input_size);

// Add biases
#pragma omp parallel for collapse(2)
  for (int j = 0; j < batch_size; j++) {
    for (int k = 0; k < output_size; k++) {
      int idx = j * output_size + k;
      output[idx] += layer->biases[k];
    }
  }

  // Apply activation function batch-wise
  activation(output, output, output_size, batch_size);
}

float *dense_forward(DenseNetwork *network, const float *input) {
  if (network == NULL || input == NULL) {
    fprintf(stderr, "Error: Null pointer passed to dense_forward\n");
    return NULL;
  }

  int batch_size = network->batch_size;
  int num_layers = network->num_layers;

  // Initial input is the provided input
  float *current_input = (float *)input;

  // Process all layers except the last one
  for (int i = 0; i < num_layers - 1; i++) {
    DenseLayer *layer = network->layers[i];
    if (layer == NULL) {
      fprintf(stderr, "Error: Null layer in network at index %d\n", i);
      return NULL;
    }

    // Process the layer with internal activation
    process_dense_layer(layer, current_input, layer->output, batch_size,
                        network->activation);

    // Update the current input to the output of this layer
    current_input = layer->output;
  }

  // Process the last layer with the output activation
  DenseLayer *last_layer = network->layers[num_layers - 1];
  if (last_layer == NULL) {
    fprintf(stderr, "Error: Null layer in network at index %d\n",
            num_layers - 1);
    return NULL;
  }

  process_dense_layer(last_layer, current_input, last_layer->output, batch_size,
                      network->output_activation);

  // Return the output of the last layer
  return last_layer->output;
}
