#include "filters.cuh"

__global__ void filters::sepiaFilterKernel(utils::byte* original, utils::byte* newImg, int size){
	int idThread = threadIdx.x, idBlock = blockIdx.x, id = idBlock * blockDim.x + idThread;

	unsigned int red = ((unsigned int)(original + id)[0]);
	unsigned int green = ((unsigned int)(original + id)[1]);
	unsigned int blue = ((unsigned int)(original + id)[2]);

	float tr = std::round( (0.393f * red) + (0.769f * green) + (0.189f * blue) );
	float tg = std::round( (0.349f * red) + (0.686f * green) + (0.168f * blue) );
	float tb = std::round( (0.272f * red) + (0.534f * green) + (0.131f * blue) ); 

	(newImg + id)[0] = (utils::byte)tr > 255 ? 255 : tr;
	(newImg + id)[1] = (utils::byte)tg > 255 ? 255 : tg;
	(newImg + id)[2] = (utils::byte)tb > 255 ? 255 : tb;
}

cudaError filters::sepiaFilterRunner(int blocks, int threads, utils::byte* original, utils::byte* newImg, int size){
	utils::byte* original_d, *newImg_d;

	cudaMalloc((void**)&original_d, size * sizeof(utils::byte));
	cudaMalloc((void**)&newImg_d, size * sizeof(utils::byte));

	cudaMemcpy(original_d, original, size * sizeof(utils::byte), cudaMemcpyHostToDevice);
	cudaMemcpy(newImg_d, newImg, size * sizeof(utils::byte), cudaMemcpyHostToDevice);

	filters::sepiaFilterKernel <<<blocks, threads>>>(original_d, newImg_d, size);

	cudaMemcpy(newImg, newImg_d, size * sizeof(utils::byte), cudaMemcpyDeviceToHost);

	return cudaGetLastError();
}