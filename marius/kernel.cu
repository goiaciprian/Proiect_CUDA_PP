#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core.hpp>

#include <stdio.h>
#include <math.h>
#include <iostream>

#include "utils.h"

#define BLOCK_SIZE 16.0

__global__ void sobelFilter(unsigned char* original, unsigned char* filterResult, const unsigned int width, const unsigned int height) {
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	float gx = 0, gy = 0;

	//printf("%d\n", (int)original[y * width + x]);
	//if ((x > 0 && x < width - 1) || (y > 0 && y < height - 1)) {
	if (x > 0 && x < width * height - 1) {

		/*gx = (-1 * original[(y - 1) * width + (x - 1)]) + 
				(-2 * original[y * width + (x - 1)]) + 
				(-1 * original[(y + 1) * width + (x - 1)]) +
				(original[(y - 1) * width + (x + 1)]) + 
				(2 * original[y * width + (x + 1)]) + 
				(original[(y + 1) * width + (x + 1)]);
		
		gy = (original[(y - 1) * width + (x - 1)]) + 
				(2 * original[(y - 1) * width + x]) + 
				(original[(y - 1) * width + (x + 1)]) +
				(-1 * original[(y + 1) * width + (x - 1)]) + 
				(-2 * original[(y + 1) * width + x]) + 
				(-1 * original[(y + 1) * width + (x + 1)]);*/

		gx = (-1 * original[(x - 1)]) +
			(-2 * original[(x - 1)]) +
			(-1 * original[(x - 1)]) +
			(original[(x + 1)]) +
			(2 * original[(x + 1)]) +
			(original[(x + 1)]);

		gy = (original[(x - 1)]) +
			(2 * original[x]) +
			(original[(x + 1)]) +
			(-1 * original[(x - 1)]) +
			(-2 * original[x]) +
			(-1 * original[(x + 1)]);

		//printf("%d\n", (int) original[y*width + x]);

		//todo aici e eroare
		filterResult[y * width + x] = (unsigned char) sqrt(gx * gx + gy * gy);
		
		//printf("%d %d %d\n", blockDim.x, blockIdx.x, threadIdx.x);
		
		//printf("%d\n",(blockDim.x + blockIdx.x) + threadIdx.x);


		//filterResult[y * width + x] = gx;
	
	}
}

__global__ void initImgArr(unsigned char* image, const unsigned int width, const unsigned int height) {
	int x = threadIdx.x + blockIdx.x + blockDim.x;
	int y = threadIdx.y + (blockIdx.y * blockDim.y);



	//if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
		image[y * width + x] = 0;
		//printf("hello from init %d\n", x);

	//}
}

__global__ void pixelsCopy(const unsigned char* original, unsigned char* filterResult, const unsigned int width, const unsigned int height) {
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);

	/*if (x >= 0 && x < width && y >= 0 && y < height) {
	}*/
		filterResult[y * width + x] = (unsigned char)(original[y * width + x]);
}

__global__ void aduna(int* a, int* b, int c) {
	int i = threadIdx.x;
	printf("%d", a[i] + b[i]);
}

int main()
{
	std::string filePath = cv::samples::findFile("sobel.png");

	cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
	cv::Mat imgGray = cv::imread(filePath, cv::IMREAD_GRAYSCALE);

	if (img.empty()) {
		return 1;
	}

	unsigned char* imgArr = utils::matToBytes(imgGray), * imgArr_d;

	cudaMalloc(&imgArr_d, img.total() * img.elemSize() * sizeof(unsigned char));
	cudaMemcpy(imgArr_d, imgArr, img.total() * img.elemSize() * sizeof(unsigned char), cudaMemcpyHostToDevice);

	//---CUDA START---
	unsigned char* filterResult_h = new unsigned char[imgGray.total() * imgGray.elemSize()]; //Host

	unsigned char* filterResult_d; //Device

	cudaMalloc((void**)&filterResult_d, imgGray.total() * imgGray.elemSize() * sizeof(unsigned char));

	cudaMemcpy(filterResult_d, filterResult_h, imgGray.total() * imgGray.elemSize() * sizeof(unsigned char), cudaMemcpyHostToDevice);

	//cudaMemset(filterResult_d, 0, imgGray.total() * imgGray.elemSize());

	//Blocks & threads
	dim3 threadsPerBlock((int)BLOCK_SIZE, (int)BLOCK_SIZE);
	dim3 blocks((int)ceil(imgGray.rows / (int)BLOCK_SIZE), (int)ceil(imgGray.cols / (int)BLOCK_SIZE));

	dim3 threads2(img.total() * img.elemSize() * sizeof(unsigned char));

	//initImgArr << < ceil(img.total() * img.elemSize()) / 1024, 1024 >> > (filterResult_d, imgGray.cols, imgGray.rows);
	sobelFilter << < ceil(img.total() * img.elemSize() / 1024), 1024>> > (imgArr_d, filterResult_d, img.cols, img.rows);
	//pixelsCopy << <ceil(img.total() * img.elemSize() / 1024), 1024 >> > (imgArr_d, filterResult_d, img.cols, img.rows);

	/*int* a = new int[2], * b = new int[2];
	a[0] = 2;
	a[1] = 4;

	b[0] = 6;
	b[1] = 12;

	int* a_d, * b_d;

	cudaMalloc(&a_d, 2 * sizeof(int));
	cudaMalloc(&b_d, 2 * sizeof(int));

	cudaMemcpy(a_d, a, 2* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, 2* sizeof(int), cudaMemcpyHostToDevice);

	aduna << <1, 2 >> > (a_d, b_d, 0);*/

	cudaError_t error = cudaGetLastError();

	if (error != cudaSuccess) {
		std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
	}

	cudaDeviceSynchronize();

	cudaMemcpy(filterResult_h, filterResult_d, imgGray.total() * imgGray.elemSize() * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	//---CUDA END---
	

	//cv::Mat mat = utils::bytesToMat(imgArr, img.cols, img.rows);
	cv::Mat sobel = utils::bytesToMat(filterResult_h, imgGray.cols, imgGray.rows);

	cv::imwrite("test.jpeg", sobel);

	cv::imshow("original", img);
	cv::imshow("originalGray", imgGray);
	cv::imshow("sobel", sobel);

	cv::waitKey(0);

	// Dimensiune imagine
	std::cout << "rows: " << imgGray.rows << " cols: " << imgGray.cols << " total: " << imgGray.total() << " elemSize: " << imgGray.elemSize() << " step[0]: " << imgGray.step[0];

	// Asa se iau valorile de RGB dintr-un pixel
	/*int x = 50, y = 30; 
	std::cout << (int)img.at<cv::Vec3b>(y, x)[0] << std::endl;
	std::cout << (int)img.at<cv::Vec3b>(y, x)[1] << std::endl;
	std::cout << (int)img.at<cv::Vec3b>(y, x)[2] << std::endl;
	*/

	//WTF
	//for (int i = 0; i < img.total() * img.elemSize(); i++) std::cout << (int)imgArr[i] + " ";

	//for (int i = 0; i < img.total() * img.elemSize(); i++) std::cout << (int)(filterResult_h+i)<<" ";

	cudaFree(&filterResult_d);
	cudaFree(&imgArr);

	return 0;
}