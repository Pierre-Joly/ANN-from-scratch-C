# Variables
CC = /opt/homebrew/opt/llvm/bin/clang
CFLAGS = -fopenmp -Rpass-analysis=loop-vectorize -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion \
         -Wnull-dereference -Wformat=2 -Wstrict-prototypes -Wmissing-prototypes -fsanitize=address \
         -fsanitize=undefined -g -Og -I./include -I./tests_include -I/opt/homebrew/opt/openblas/include

LDFLAGS = -L/opt/homebrew/opt/openblas/lib -lopenblas

SRC_DIR = src
TEST_DIR = tests
OBJ_DIR = obj
BIN_DIR = bin
TARGET = $(BIN_DIR)/project

# Production files
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

# Test files
TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)
TEST_OBJS = $(patsubst $(TEST_DIR)/%.c, $(OBJ_DIR)/tests/%.o, $(TEST_SRCS))

# Build Rules
all: $(TARGET)

# Rule to build the main target
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to build object files for production sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to build object files for test sources
$(OBJ_DIR)/tests/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to build the tests executable
tests: $(TEST_OBJS) $(filter-out $(OBJ_DIR)/main.o, $(OBJS))
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/tests $^ $(LDFLAGS)

# Rule to run the tests
run-tests: tests
	@echo "Running tests..."
	./$(BIN_DIR)/tests

# Rule to clean up the build directories
clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean tests run-tests
