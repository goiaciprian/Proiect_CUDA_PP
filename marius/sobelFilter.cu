#include "filters.cuh"


__global__ void filters::sobelFilterKernel(utils::byte* original, utils::byte* filterResult, int , unsigned int width, unsigned int height) {
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	float gx = 0, gy = 0;

	if (x > 0 && x < width * height - 1) {


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

		filterResult[y * width + x] = (utils::byte)sqrt(gx * gx + gy * gy);
	}
}


cudaError filters::sobelFilterRunner(int blocks, int threads, utils::byte* original_h, utils::byte* filterResult_h, int size, unsigned int width, unsigned int height) {
	utils::byte* original_d, *filterResult_d;

	cudaMalloc((void**) & original_d, size * sizeof(utils::byte));
	cudaMalloc((void**) & filterResult_d, size * sizeof(utils::byte));

	cudaMemcpy(original_d, original_h, size * sizeof(utils::byte), cudaMemcpyHostToDevice);
	cudaMemcpy(filterResult_d, filterResult_h, size * sizeof(utils::byte), cudaMemcpyHostToDevice);

	filters::sobelFilterKernel << <blocks, threads >> > (original_d, filterResult_d, size, width, height);

	cudaMemcpy(filterResult_h, filterResult_d, size * sizeof(utils::byte), cudaMemcpyDeviceToHost);

	cudaFree(&original_d);
	cudaFree(&filterResult_d);


	return cudaGetLastError();
}