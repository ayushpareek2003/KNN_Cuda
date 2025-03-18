# K-Nearest Neighbors (KNN) Implementation on CUDA

## Overview
This repository contains an implementation of the K-Nearest Neighbors (KNN) algorithm in C++ using CUDA. The implementation includes two different kernel versions:
1. **Standard KNN Kernel**: A straightforward parallel implementation of KNN.
2. **Shared Memory Optimized KNN Kernel**: A more efficient version utilizing CUDA's shared memory to reduce global memory access latency and improve performance.

## Repository Structure
```
├── inc/                # Header files
├── src/                # Source files (KNN implementation and main logic)
├── main.cu             # Main program
├── Test.cu             # Testing program
├── CMakeLists.txt      # CMake configuration file
├── test.txt            # Sample test dataset
├── train.txt           # Sample train dataset


```

## Features
- Implements KNN algorithm in CUDA for parallel processing
- Includes two kernels: a basic one and an optimized one using shared memory
- Modular structure with separate headers and source files

## Installation
### Prerequisites
- CUDA Toolkit (Tested on CUDA 11+)
- CMake (for building the project)
- NVIDIA GPU with CUDA support
- g++ or MSVC compiler

### Build Instructions
1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Create a build directory and compile the project:
   ```sh
   mkdir build && cd build
   cmake ..
   make
   ```
3. Run the executable:
   ```sh
   ./knn_cuda
   ```

## Usage
Modify `test.txt` to include the dataset you want to use. The program will read input from `test.txt` and process KNN classification using CUDA

## Performance Comparison
The shared memory-optimized kernel significantly reduces memory access latency and improves execution speed compared to the standard kernel. This is especially noticeable with larger datasets where memory access patterns become a bottleneck

## Future Improvements
- Implementing an efficient distance metric calculation
- Adding support for k-d tree acceleration
- Optimizing memory transfers between host and device

## Contributing
Feel free to fork the repository and submit pull requests for improvements



