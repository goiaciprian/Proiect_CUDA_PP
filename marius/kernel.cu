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
	std::string filePath = cv::samples::findFile("thisisdog.jpeg");
	cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);

	int size = img.total() * img.elemSize();
	int threads = 1024;
	int blocks = ceil(size / threads);

	utils::byte *imagine = utils::matToBytes(img);
	utils::byte *newImage_grey = new utils::byte[size];
	utils::byte* newImage_inverted = new utils::byte[size];


	cudaError status = filters::greyFilterRunner(blocks, threads, imagine, newImage_grey, size);

	if (status != cudaSuccess) {
		std::cout << "Eroare cuda: " << cudaGetErrorString(status) << std::endl;
	}

	cudaError status_invert = filters::invertFilterRunner(blocks, threads, imagine, newImage_inverted, size);
	
	if (status_invert != cudaSuccess) {
		std::cout << "Eroare cuda: " << cudaGetErrorString(status) << std::endl;
	}


	cv::Mat newimg_grey = utils::bytesToMat(newImage_grey, img.cols, img.rows);
	cv::Mat newimg_invert = utils::bytesToMat(newImage_inverted, img.cols, img.rows);


	cv::imshow("Original", img);
	cv::imshow("Greyfilter", newimg_grey);
	cv::imshow("Inverted", newimg_invert);

	int test = cv::waitKey();


	return 0;
}