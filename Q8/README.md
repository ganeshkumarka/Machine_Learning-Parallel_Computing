# Parallel Image Processing with OpenMP

## Introduction to OpenMP

OpenMP (Open Multi-Processing) is an API that supports multi-platform shared memory multiprocessing programming in C, C++, and Fortran. It provides a simple and flexible interface for developing parallel applications.

### Key Concepts

- **Parallelism**: The process of executing multiple tasks simultaneously to improve performance.
- **Threads**: The smallest unit of processing that can be scheduled by an operating system.
- **Parallel Regions**: Blocks of code that are executed by multiple threads in parallel.

## Loading Image

We use OpenCV to load an image from a file into a matrix representation. OpenCV is a powerful library for image processing and computer vision tasks.

## Image Processing Operations

We implement the following image processing operations as matrix operations:

1. **Image Blurring**: Apply a blur filter to the image using a convolution matrix/kernel.
2. **Image Sharpening**: Apply a sharpening filter to the image using a convolution matrix/kernel.
3. **Image Edge Detection**: Apply an edge detection filter (e.g., Sobel operator) to detect edges in the image.

Each of these operations is parallelized using OpenMP directives to exploit parallelism.

## Performance Analysis

We measure the execution time of each image processing operation with and without parallelization, compare the performance improvement achieved by parallelizing the operations using OpenMP, and analyze the speedup and efficiency achieved by parallelization.

## Visualization and Output

We display the processed images after each image processing operation for visualization and save the processed images to files for further analysis and comparison.

## Running the Code

To compile and run the code:

```bash
g++ -fopenmp -o parallel_image_processing parallel_image_processing.cpp `pkg-config --cflags --libs opencv4`
./parallel_image_processing
```

## Requirements

- OpenCV
- OpenMP
- C++ Compiler (e.g., g++)

Install dependencies with:

```bash
sudo apt-get install libopencv-dev
```
