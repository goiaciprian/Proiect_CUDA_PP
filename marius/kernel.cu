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
	while(1) {
		std::string fileName = "";
		int input = 0;

		std::cout << "Alegeti poza:\n1.Bucsa\n2.Hippo \n3.Base\n4.Sobel\n5.Creation\n6.Starea noastra mentala\n7.NFT \n\nAlegere: ";
		std::cin >> input;

		switch(input) {
		case 1:
			fileName = "bucsa.jpg";
			break;
		case 2:
			fileName = "jordan.jpeg";
			break;
		case 3:
			fileName = "base.jpg";
			break;
		case 4:
			fileName = "sobel.png";
			break;
		case 5:
			fileName = "creation.jpeg";
			break;
		case 6:
			fileName = "starementala.jpeg";
			break;
		case 7:
			fileName = "nft.jpeg";
			break;

		default:
			return;
		}

		system("cls");

		std::string filePath = cv::samples::findFile(fileName);

		cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
	
		int size = img.total() * img.elemSize();
		int threads = 1024;
		int blocks = ceil(size / threads);

		utils::byte *imagine = utils::matToBytes(img);

		utils::byte *newImage_grey = new utils::byte[size];
		utils::byte* newImage_inverted = new utils::byte[size];
		utils::byte* newImage_sobel = new utils::byte[img.total()];
		utils::byte* newImage_sepia = new utils::byte[size];

		cudaError status_grey = filters::greyFilterRunner(blocks, threads, imagine, newImage_grey, size);

		if (status_grey != cudaSuccess) {
			std::cout << "Eroare cuda grey : " << cudaGetErrorString(status_grey) << std::endl;
		}
		cudaError status_invert = filters::invertFilterRunner(blocks, threads, imagine, newImage_inverted, size);
	
		if (status_invert != cudaSuccess) {
			std::cout << "Eroare cuda invert: " << cudaGetErrorString(status_invert) << std::endl;
		}

		utils::byte* grey1Channel = utils::ch3toCh1(newImage_grey, img.total(), size);


		cudaError status_sobel = filters::sobelFilterRunner(blocks, threads, grey1Channel, newImage_sobel, img.total(), img.rows, img.cols);

		if (status_sobel != cudaSuccess) {
			std::cout << "Eroare cuda: " << cudaGetErrorString(status_sobel) << std::endl;
		}

		cudaError status_sepia = filters::sepiaFilterRunner(blocks, threads, imagine, newImage_sepia, size);
		if (status_sepia != cudaSuccess) {
			std::cout << "Eroare cuda: " << cudaGetErrorString(status_sobel) << std::endl;
		}


		cv::Mat newimg_grey = utils::bytesToMat(grey1Channel, img.cols, img.rows, CV_8U);
		cv::Mat newimg_invert = utils::bytesToMat(newImage_inverted, img.cols, img.rows, CV_8UC3);
		cv::Mat newimg_sobel = utils::bytesToMat(newImage_sobel, img.cols, img.rows, CV_8U);
		cv::Mat newimg_sepia = utils::bytesToMat(newImage_sepia, img.cols, img.rows, CV_8UC3);

		cv::imshow("Original", img);
		cv::imshow("Greyfilter", newimg_grey);
		cv::imshow("Inverted", newimg_invert);
		cv::imshow("Sobel", newimg_sobel);
		cv::imshow("Drog", newimg_sepia);


		int test = cv::waitKey();
		cv::destroyAllWindows();
		system("cls");
	}

	return 0;
}