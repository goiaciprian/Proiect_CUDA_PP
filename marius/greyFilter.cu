#include "filters.cuh"


__global__ void filters::greyFilterKernel(utils::byte *image, utils::byte* newImage, int size) {
	int idThread = threadIdx.x, idBlock = blockIdx.x, id = idBlock * blockDim.x + idThread;

	unsigned int red = (unsigned int)(image + id)[0];
	unsigned int green = (unsigned int)(image + id)[1];
	unsigned int blue = (unsigned int)(image + id)[2];

	newImage[id] = (red + green + blue) / 3;
}


cudaError filters::greyFilterRunner(int blocks, int threads, utils::byte* image, utils::byte* newImage, int size) {
	utils::byte* originalImage;

	utils::byte* newImage_d;

	cudaMalloc((void**)&originalImage, size * sizeof(utils::byte));
	cudaMalloc((void**)&newImage_d, size * sizeof(utils::byte));

	cudaMemcpy(originalImage, image, size * sizeof(utils::byte), cudaMemcpyHostToDevice);
	cudaMemcpy(newImage_d, newImage, size * sizeof(utils::byte), cudaMemcpyHostToDevice);

	filters::greyFilterKernel << < blocks, threads >> > (originalImage, newImage_d, size);

	cudaMemcpy(newImage, newImage_d, size * sizeof(utils::byte), cudaMemcpyDeviceToHost);

	cudaFree(&originalImage);
	cudaFree(&newImage_d);

	return cudaGetLastError();
};
