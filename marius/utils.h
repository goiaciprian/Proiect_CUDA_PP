#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace utils {
	uint8_t* matToBytes(cv::Mat& image);
	cv::Mat bytesToMat(uint8_t* bytes, int width, int height);
}