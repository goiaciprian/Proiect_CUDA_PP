
#include "utils.h"

utils::byte* utils::matToBytes(cv::Mat& image) {
	int size = image.total() * image.elemSize();
	utils::byte* bytes = new utils::byte[size];
	std::memcpy(bytes, image.data, size * sizeof(utils::byte));
	return bytes;
}

cv::Mat utils::bytesToMat(utils::byte *bytes, int width, int heigth) {
	cv::Mat image = cv::Mat(heigth, width, CV_8UC3, bytes).clone();
	return image;
}