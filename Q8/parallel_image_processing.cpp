#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

// Function to load an image from a file into a matrix representation
Mat loadImage(const string &filename)
{
    Mat image = imread(filename, IMREAD_COLOR);
    if (image.empty())
    {
        cerr << "Error: Could not open or find the image!" << endl;
        exit(EXIT_FAILURE);
    }
    return image;
}

// Function to apply a convolution filter to an image
Mat applyFilter(const Mat &image, const Mat &kernel)
{
    Mat result;
    filter2D(image, result, -1, kernel);
    return result;
}

// Function to apply a convolution filter to an image in parallel using OpenMP
Mat applyFilterParallel(const Mat &image, const Mat &kernel)
{
    Mat result = image.clone();
    int rows = image.rows;
    int cols = image.cols;
    int kRows = kernel.rows;
    int kCols = kernel.cols;
    int kCenterX = kCols / 2;
    int kCenterY = kRows / 2;

#pragma omp parallel for collapse(2)
    for (int i = kCenterY; i < rows - kCenterY; ++i)
    {
        for (int j = kCenterX; j < cols - kCenterX; ++j)
        {
            Vec3f sum = Vec3f(0, 0, 0);
            for (int m = 0; m < kRows; ++m)
            {
                for (int n = 0; n < kCols; ++n)
                {
                    int x = j + n - kCenterX;
                    int y = i + m - kCenterY;
                    sum += kernel.at<float>(m, n) * image.at<Vec3b>(y, x);
                }
            }
            result.at<Vec3b>(i, j) = sum;
        }
    }
    return result;
}

int main()
{
    // Load the image
    string filename = "Q8/cusat.png";
    Mat image = loadImage(filename);
    imshow("Original Image", image);
    waitKey(0);

    // Define convolution kernels
    Mat blurKernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1) / 9.0;
    Mat sharpenKernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    Mat edgeKernel = (Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);

    // Apply filters sequentially
    double start = omp_get_wtime();
    Mat blurredImage = applyFilter(image, blurKernel);
    double end = omp_get_wtime();
    cout << "Blurring Time (Sequential): " << (end - start) << " seconds" << endl;
    imshow("Blurred Image", blurredImage);
    waitKey(0);

    start = omp_get_wtime();
    Mat sharpenedImage = applyFilter(image, sharpenKernel);
    end = omp_get_wtime();
    cout << "Sharpening Time (Sequential): " << (end - start) << " seconds" << endl;
    imshow("Sharpened Image", sharpenedImage);
    waitKey(0);

    start = omp_get_wtime();
    Mat edgeImage = applyFilter(image, edgeKernel);
    end = omp_get_wtime();
    cout << "Edge Detection Time (Sequential): " << (end - start) << " seconds" << endl;
    imshow("Edge Detected Image", edgeImage);
    waitKey(0);

    // Apply filters in parallel
    start = omp_get_wtime();
    Mat blurredImageParallel = applyFilterParallel(image, blurKernel);
    end = omp_get_wtime();
    cout << "Blurring Time (Parallel): " << (end - start) << " seconds" << endl;
    imshow("Blurred Image (Parallel)", blurredImageParallel);
    waitKey(0);

    start = omp_get_wtime();
    Mat sharpenedImageParallel = applyFilterParallel(image, sharpenKernel);
    end = omp_get_wtime();
    cout << "Sharpening Time (Parallel): " << (end - start) << " seconds" << endl;
    imshow("Sharpened Image (Parallel)", sharpenedImageParallel);
    waitKey(0);

    start = omp_get_wtime();
    Mat edgeImageParallel = applyFilterParallel(image, edgeKernel);
    end = omp_get_wtime();
    cout << "Edge Detection Time (Parallel): " << (end - start) << " seconds" << endl;
    imshow("Edge Detected Image (Parallel)", edgeImageParallel);
    waitKey(0);

    // Save processed images
    imwrite("blurred_image.jpg", blurredImage);
    imwrite("sharpened_image.jpg", sharpenedImage);
    imwrite("edge_image.jpg", edgeImage);
    imwrite("blurred_image_parallel.jpg", blurredImageParallel);
    imwrite("sharpened_image_parallel.jpg", sharpenedImageParallel);
    imwrite("edge_image_parallel.jpg", edgeImageParallel);

    return 0;
}
