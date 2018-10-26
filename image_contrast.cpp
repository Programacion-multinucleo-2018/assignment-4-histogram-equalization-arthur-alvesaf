// Arthur Alves Araujo Ferreira
// A01022593
// Compile with g++ -o contrast image_contrast.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc

// Includes
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

// Histogram equalization function given an opencv mat and output
void fix_contrast_image(const cv::Mat& input, cv::Mat& output)
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
	fix_contrast_image(input_gray, output);

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
