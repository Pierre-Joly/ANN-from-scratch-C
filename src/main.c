#include <stdio.h>

#include "activations.h"
#include "forward.h"
#include "networks.h"

int main(void) {
  // Define dimensions
  int input_size = 4;   // Input dimension
  int batch_size = 2;   // Number of samples in a batch
  int hidden_size = 5;  // Hidden layer dimension
  int output_size = 3;  // Output dimension

  // Create the network
  DenseNetwork *network =
      create_dense_network(2, batch_size, relu_batch, softmax_batch);
  if (network == NULL) {
    fprintf(stderr, "Failed to create network\n");
    return 1;
  }

  // Add layers to the network
  DenseLayer *layer1 = create_dense_layer(input_size, hidden_size, batch_size);
  DenseLayer *layer2 = create_dense_layer(hidden_size, output_size, batch_size);

  if (layer1 == NULL || layer2 == NULL) {
    fprintf(stderr, "Failed to create layers\n");
    free_dense_network(network);
    return 1;
  }

  if (add_layer(network, 0, layer1) != 0 ||
      add_layer(network, 1, layer2) != 0) {
    fprintf(stderr, "Failed to add layers to network\n");
    free_dense_network(network);
    return 1;
  }

  // Define input (batch of 2 samples, each with 4 features)
  float input[8] = {0.5f, 1.0f, -0.5f, 2.0f, 0.1f, -1.5f, 3.0f, -0.8f};

  // Perform forward pass
  float *output = dense_forward(network, input);
  if (output == NULL) {
    fprintf(stderr, "Error during forward pass\n");
    free_dense_network(network);
    return 1;
  }

  // Print the output
  printf("Network output:\n");
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < output_size; j++) {
      printf("%f ", output[i * output_size + j]);
    }
    printf("\n");
  }

  // Free resources
  free_dense_network(network);
  return 0;
}
