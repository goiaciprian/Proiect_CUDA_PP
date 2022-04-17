
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "utils.h"

uint8_t * utils::matToBytes(cv::Mat& image) {
	int size = image.total() * image.elemSize();
	uint8_t* bytes = new uint8_t[size];
	std::memcpy(bytes, image.data, size * sizeof(uint8_t));
	return bytes;
}

cv::Mat utils::bytesToMat(uint8_t* bytes, int width, int heigth) {
	cv::Mat image = cv::Mat(heigth, width, CV_8UC3, bytes).clone();
	return image;
}