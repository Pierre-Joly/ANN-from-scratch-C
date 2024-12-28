# Variables
CC = /opt/homebrew/opt/llvm/bin/clang
CFLAGS = -fopenmp -Rpass-analysis=loop-vectorize -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wnull-dereference \
         -Wformat=2 -Wstrict-prototypes -Wmissing-prototypes -fsanitize=address \
         -fsanitize=undefined -g -Og -I./include -I./tests_include
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

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/tests/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

tests: $(TEST_OBJS) $(filter-out $(OBJ_DIR)/main.o, $(OBJS))
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/tests $^

run-tests: tests
	@echo "Running tests..."
	./$(BIN_DIR)/tests

clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean tests run-tests
