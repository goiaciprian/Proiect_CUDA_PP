#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>

#include "utils.h"
#include "filters.cuh"


int main()
{
	std::string filePath = cv::samples::findFile("sobel.png");

	cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
	cv::Mat imgGreyscale = cv::imread(filePath, cv::IMREAD_GRAYSCALE);

	int size = img.total() * img.elemSize();
	int threads = 1024;
	int blocks = ceil(size / threads);

	utils::byte* forSobel = utils::matToBytes(imgGreyscale);

	utils::byte *imagine = utils::matToBytes(img);
	utils::byte *newImage_grey = new utils::byte[size];
	utils::byte* newImage_inverted = new utils::byte[size];
	utils::byte* newImage_sobel = new utils::byte[size];

	cudaError status_grey = filters::greyFilterRunner(blocks, threads, imagine, newImage_grey, size);

	if (status_grey != cudaSuccess) {
		std::cout << "Eroare cuda: " << cudaGetErrorString(status_grey) << std::endl;
	}
	cudaError status_invert = filters::invertFilterRunner(blocks, threads, imagine, newImage_inverted, size);
	
	if (status_invert != cudaSuccess) {
		std::cout << "Eroare cuda: " << cudaGetErrorString(status_invert) << std::endl;
	}

	cudaError status_sobel = filters::sobelFilterRunner(blocks, threads, forSobel, newImage_sobel, size, img.rows, img.cols);

	if (status_sobel != cudaSuccess) {
		std::cout << "Eroare cuda: " << cudaGetErrorString(status_sobel) << std::endl;
	}

	std::cout << CV_8U;

	cv::Mat newimg_grey = utils::bytesToMat(newImage_grey, img.cols, img.rows, CV_8UC3);
	cv::Mat newimg_invert = utils::bytesToMat(newImage_inverted, img.cols, img.rows, CV_8UC3);
	cv::Mat newimg_sobel = utils::bytesToMat(newImage_sobel, img.cols, img.rows, CV_8U);


	cv::imshow("Original", img);
	cv::imshow("Greyfilter", newimg_grey);
	cv::imshow("Inverted", newimg_invert);
	cv::imshow("Sobel", newimg_sobel);

	int test = cv::waitKey();


	return 0;
}