# ANN-from-scratch-C

**Educational project** to implement a feedforward Neural Network (NN) entirely in **C**, with minimal dependencies (just BLAS for matrix multiplication). The purpose is to gain a deeper understanding of how neural networks work under the hood by coding every major component—forward pass, backpropagation, and more—manually.

---

## Table of Contents

- [ANN-from-scratch-C](#ann-from-scratch-c)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dependencies](#dependencies)
  - [Building and Running](#building-and-running)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Build](#2-build)
    - [3. Run](#3-run)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)

---

## Introduction

This project is not intended as a fully optimized deep learning framework. Instead, it focuses on education and comprehension, showing how to:

- Load a small dataset from a file.  
- Initialize network parameters (weights, biases).  
- Perform matrix multiplications using BLAS (`cblas_sgemm`).  
- Implement forward propagation with customizable activation functions.  
- Implement backpropagation with basic gradient descent.  
- Tweak hyperparameters such as number of layers, neurons per layer, learning rate, etc.

By reading and modifying the code, you’ll get hands-on experience with how feedforward neural networks operate at a low level.

---

## Dependencies

1. **C Compiler**  
   - Tested on `clang`.
2. **BLAS Library**  
   - For matrix multiplication (`-lblas` or `-lcblas` linking).
3. **Make**

---

## Building and Running

### 1. Clone the Repository
```bash
git clone https://github.com/Pierre-Joly/ANN-from-scratch-C.git
cd ANN-from-scratch-C
```

### 2. Build

- **Using `make`**  
  ```bash
  make
  ```
  This produces an executable (e.g., `ANN-from-scratch-C.out`) in the same directory.

### 3. Run
```bash
./ANN-from-scratch-C.out
```
Or if you used `gcc -o ann`:
```bash
./ann
```

By default, the network will read the dataset from `data/dataset.txt`, train for a preset number of epochs, then print out its performance.

---

## Project Structure

```
ANN-from-scratch-C/
├── data/
│   └── dataset.txt         # Default dataset used for training
├── doc/
│   ├── ANN-from-scratch.pdf # PDF doc with more details
│   └── Doxyfile            # Doxygen config for generating HTML doc
├── dataset.c
├── dataset.h
├── main.c
├── net.c
├── net.h
├── Makefile
└── README.md
```

## Contributing

Contributions, bug reports, and feature requests are welcome! To contribute:

1. **Fork** this repository.  
2. **Create a new branch** for your feature or fix:
   ```bash
   git checkout -b feature-my-awesome-feature
   ```
3. **Commit** your changes:
   ```bash
   git commit -m "Implement new activation function"
   ```
4. **Push** to your branch:
   ```bash
   git push origin feature-my-awesome-feature
   ```
5. **Open a Pull Request** on GitHub.

---

## License

This project is licensed under the [MIT License](LICENSE).