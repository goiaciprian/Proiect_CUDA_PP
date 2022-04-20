#include "filters.cuh"

__global__ void filters::invertFilterKernel(utils::byte* original, utils::byte* newImg, int size) {
	int idThread = threadIdx.x, idBlock = blockIdx.x, id = idBlock * blockDim.x + idThread;

	unsigned int red = 255 - ((unsigned int)(original + id)[0]);
	unsigned int green = 255 - ((unsigned int)(original + id)[1]);
	unsigned int blue = 255 - ((unsigned int)(original + id)[2]);

	(newImg + id)[0] = (utils::byte)red;
	(newImg + id)[1] = (utils::byte)green;
	(newImg + id)[2] = (utils::byte)blue;
}


cudaError filters::invertFilterRunner(int blocks, int threads, utils::byte* original, utils::byte* newImg, int size) {

	utils::byte* originalImg_d;
	utils::byte* newImage_d;

	cudaMalloc((void**)&originalImg_d, size * sizeof(utils::byte));
	cudaMalloc((void**)&newImage_d, size * sizeof(utils::byte));

	cudaMemcpy(originalImg_d, original, size * sizeof(utils::byte), cudaMemcpyHostToDevice);
	cudaMemcpy(newImage_d, newImg, size * sizeof(utils::byte), cudaMemcpyHostToDevice);

	filters::invertFilterKernel << <blocks, threads >> > (originalImg_d, newImage_d, size);

	cudaMemcpy(newImg, newImage_d, size * sizeof(utils::byte), cudaMemcpyDeviceToHost);

	cudaFree(&originalImg_d);
	cudaFree(&newImage_d);

	return cudaGetLastError();

}