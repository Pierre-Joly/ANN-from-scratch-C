#include <stdio.h>

#include "test_activations.h"
#include "test_forward.h"
#include "test_layers.h"
#include "test_loss.h"
#include "test_networks.h"
#include "test_operations.h"

int main(void) {
  printf("Running tests...\n");

  test_activations();
  test_forward();
  test_layers();
  test_loss();
  test_networks();
  test_operations();

  printf("All tests passed!\n");
  return 0;
}
