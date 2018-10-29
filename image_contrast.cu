// Arthur Alves Araujo Ferreira
// A01022593
// Compile with nvcc -o contrast image_contrast.cu -lopencv_core -lopencv_highgui -lopencv_imgproc

// Includes
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>

using namespace std;

__global__ void fill_histogram(unsigned char *input, int width, int height, int *histogram) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * width + ix;

  if (ix < width && iy < height)
    atomicAdd(&histogram[ (int)input[idx] ], 1);
}

void equalize_histogram(int *histogram, int *normalized_histogram, int width, int height) {
  unsigned long int accumulated = 0;
  for (int i=0; i<256; i++) {
    accumulated += histogram[i];
    normalized_histogram[i] = accumulated * 255 / (width * height);
  }
}

__global__ void equalize_image(unsigned char *input, unsigned char *output, int width, int height, int *normalized_histogram) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * width + ix;

  if (ix < width && iy < height)
    output[idx] = normalized_histogram[ (int)input[idx] ];
}

// Histogram equalization function given an opencv mat and output CPU
void fix_contrast_image_cpu(const cv::Mat& input, cv::Mat& output)
{
	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	int histogram[256] = {0};

	// Count pixel occurence on histogram
  for (int pixel_tid = 0; pixel_tid < input.cols*input.rows; pixel_tid++) {
    histogram[ (int)input.data[pixel_tid] ] ++;
  }

  // Normalize histogram with cumulative distribution function and set range from 0 to 255
  unsigned long int normalized_histogram[256];
	unsigned long int accumulated = 0;
	for (int i=0; i<256; i++) {
		accumulated += histogram[i];
		normalized_histogram[i] = accumulated * 255 / (input.cols*input.rows);
	}

  // Set resulting pixel values to output
	for (int pixel_tid = 0; pixel_tid < input.cols*input.rows; pixel_tid++) {
    output.data[pixel_tid] = normalized_histogram[ (int)input.data[pixel_tid] ];
  }
}

void fix_contrast_image_gpu(const cv::Mat &input, cv::Mat &output) {
  // Set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  cout << "Using Device "<< dev << deviceProp.name << endl;
  cudaSetDevice(dev);

  cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
  const dim3 block (32,32);
  const dim3 grid ( (int)ceil((input.cols) / block.x),
  (int)ceil((input.rows) / block.y));

  const int bytes = input.rows * input.cols;
  int hist_bytes = sizeof(int) * 256;

  unsigned char *d_input, *d_output;

  cudaMalloc<unsigned char>(&d_input, bytes);
  cudaMalloc<unsigned char>(&d_output, bytes);

  cudaMemcpy(d_input, input.ptr(), bytes, cudaMemcpyHostToDevice);

  int histogram[256] = {0};
  int *d_hist;
  cudaMalloc((void **)&d_hist, hist_bytes);
  int normalized_histogram[256] = {0};
  int *dn_hist;
  cudaMalloc((void **)&dn_hist, hist_bytes);
  // cudaMemset(d_hist, 0, hist_bytes);

  fill_histogram <<<grid,block>>> (d_input, input.cols, input.rows, d_hist);

  cudaMemcpy(histogram, d_hist, hist_bytes, cudaMemcpyDeviceToHost);
  equalize_histogram(histogram, normalized_histogram, input.cols, input.rows);
  cudaMemcpy(dn_hist, normalized_histogram, hist_bytes, cudaMemcpyHostToDevice);

  equalize_image <<<grid, block>>> (d_input, d_output, input.cols, input.rows, dn_hist);
  cudaMemcpy(output.ptr(), d_output, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_hist);
  cudaFree(dn_hist);
}

// Main function
int main(int argc, char *argv[])
{
	// Variable initialization
	string imagePath;

	// Check for program inputs
	if(argc < 2)
		imagePath = "Images/dog1.jpeg";
	else
		imagePath = argv[1];

	// Read input image from the disk (color)
	cv::Mat input_color = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input_color.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

  // Convert image from color to grayscale
	cv::Mat input_gray;
  cvtColor(input_color, input_gray, cv::COLOR_BGR2GRAY);

	//Create output image
  cv::Mat output(input_gray.rows, input_gray.cols, input_gray.type());

	//Call the image manipulation function
  // fix_contrast_image_cpu(input_gray, output);
	fix_contrast_image_gpu(input_gray, output);

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	cv::imshow("Input", input_gray);
	cv::imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
