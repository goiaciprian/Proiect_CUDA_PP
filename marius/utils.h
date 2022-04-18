#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace utils {
	unsigned char* matToBytes(cv::Mat& image);
	cv::Mat bytesToMat(unsigned char* bytes, int width, int height);
}