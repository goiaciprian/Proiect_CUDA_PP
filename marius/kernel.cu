#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core.hpp>

#include <stdio.h>
#include <math.h>
#include <iostream>

#include "utils.h"

int main()
{
	std::string filePath = cv::samples::findFile("thisisdog.jpeg");
	cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);

	if (img.empty()) {
		return 1;
	}

	uint8_t* imgArr = utils::matToBytes(img);

	cv::Mat mat = utils::bytesToMat(imgArr, img.cols, img.rows);

	cv::imwrite("test.jpeg", mat);
   
}