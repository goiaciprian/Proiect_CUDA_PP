#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace utils {
	typedef unsigned char byte;

	byte* matToBytes(cv::Mat& image);
	
	cv::Mat bytesToMat(byte* bytes, int width, int height, int CVTYPE);
}