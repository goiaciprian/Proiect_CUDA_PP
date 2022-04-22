#include "utils.h"

utils::byte* utils::matToBytes(cv::Mat& image) {
	int size = image.total() * image.elemSize();
	utils::byte* bytes = new utils::byte[size];
	std::memcpy(bytes, image.data, size * sizeof(utils::byte));
	return bytes;
};

cv::Mat utils::bytesToMat(utils::byte* bytes, int width, int height, int CVTYPE) {
	cv::Mat image = cv::Mat(height, width, CVTYPE,  bytes).clone();
	return image;
};

utils::byte* utils::ch3toCh1(utils::byte* bytes, int sizeGrey, int size) {
	utils::byte* ch1Arr = new utils::byte[sizeGrey];
	for (int i = 0, j = 0; i < size; i += 3, j++) {
		ch1Arr[j] = bytes[i];
	}
	return ch1Arr;
}