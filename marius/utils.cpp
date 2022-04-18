
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "utils.h"

unsigned char * utils::matToBytes(cv::Mat& image) {
	int size = image.total() * image.elemSize();
	unsigned char* bytes = new unsigned char[size];
	std::memcpy(bytes, image.data, size * sizeof(unsigned char));
	return bytes;
}

cv::Mat utils::bytesToMat(unsigned char* bytes, int width, int height) {
	cv::Mat image = cv::Mat(height, width, CV_8U, bytes).clone();
	return image;
}