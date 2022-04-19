#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"

#include <stdio.h>
#include <iostream>


namespace filters{
	__global__ void greyFilterKernel(utils::byte* img, utils::byte *newImage, int size);
	cudaError greyFilterRunner(int blocks, int threads, utils::byte* image, utils::byte* newImage, int size);

	__global__ void invertFilterKernel(utils::byte* img, utils::byte* newImg, int size);
	cudaError invertFilterRunner(int blocks, int threads, utils::byte* image, utils::byte* newImage, int size);

}