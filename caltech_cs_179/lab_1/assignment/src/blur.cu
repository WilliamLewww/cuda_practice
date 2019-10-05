/* 
 * CUDA blur
 * Kevin Yuh, 2014 
 * Revised by Nailen Matschke, 2016
 * Revised by Loko Kung, 2018
 */
#include "blur.cuh"

#include <cstdio>
#include <cuda_runtime.h>

#include "cuda_header.cuh"

CUDA_CALLABLE
void cuda_blur_kernel_convolution(uint thread_index, const float* gpu_raw_data,
                                  const float* gpu_blur_v, float* gpu_out_data,
                                  const unsigned int n_frames,
                                  const unsigned int blur_v_size) {
    // TODO: Implement the necessary convolution function that should be
    //       completed for each thread_index. Use the CPU implementation in
    //       blur.cpp as a reference.

    if (thread_index < blur_v_size) {
        for (int x = 0; x <= thread_index; x++) {
            gpu_out_data[thread_index] += gpu_raw_data[thread_index - x] * gpu_blur_v[x];
        }
    }
    if (thread_index >= blur_v_size) {
        for (int x = 0; x <= blur_v_size; x++) {
            gpu_out_data[thread_index] += gpu_raw_data[thread_index - x] * gpu_blur_v[x];
        }
    }
}

__global__
void cuda_blur_kernel(const float *gpu_raw_data, const float *gpu_blur_v,
                      float *gpu_out_data, int n_frames, int blur_v_size) {
    // TODO: Compute the current thread index.
    uint thread_index = (blockIdx.x * blockDim.x) + threadIdx.x;

    // TODO: Update the while loop to handle all indices for this thread.
    //       Remember to advance the index as necessary.
    while (thread_index < n_frames + blur_v_size) {
        // Do computation for this thread index
        cuda_blur_kernel_convolution(thread_index, gpu_raw_data,
                                     gpu_blur_v, gpu_out_data,
                                     n_frames, blur_v_size);
        // TODO: Update the thread index
        thread_index += blockDim.x * gridDim.x;
    }
}

float cuda_call_blur_kernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            const float *raw_data,
                            const float *blur_v,
                            float *out_data,
                            const unsigned int n_frames,
                            const unsigned int blur_v_size) {
    // Use the CUDA machinery for recording time
    cudaEvent_t start_gpu, stop_gpu;
    float time_milli = -1;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    // TODO: Allocate GPU memory for the raw input data (either audio file
    //       data or randomly generated data. The data is of type float and
    //       has n_frames elements. Then copy the data in raw_data into the
    //       GPU memory you allocated.
    float* gpu_raw_data;
    cudaMalloc((float**)&gpu_raw_data, n_frames*sizeof(float));
    cudaMemcpy(gpu_raw_data, raw_data, n_frames*sizeof(float), cudaMemcpyHostToDevice);

    // TODO: Allocate GPU memory for the impulse signal (for now global GPU
    //       memory is fine. The data is of type float and has blur_v_size
    //       elements. Then copy the data in blur_v into the GPU memory you
    //       allocated.
    float* gpu_blur_v;
    cudaMalloc((float**)&gpu_blur_v, blur_v_size*sizeof(float));
    cudaMemcpy(gpu_blur_v, blur_v, blur_v_size*sizeof(float), cudaMemcpyHostToDevice);

    // TODO: Allocate GPU memory to store the output audio signal after the
    //       convolution. The data is of type float and has n_frames elements.
    //       Initialize the data as necessary.
    float* gpu_out_data;
    cudaMalloc((float**)&gpu_out_data, n_frames*sizeof(float));
    
    // TODO: Appropriately call the kernel function.
    cuda_blur_kernel<<<blocks, threads_per_block>>>(gpu_raw_data, gpu_blur_v, gpu_out_data, n_frames, blur_v_size);

    // Check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    else
        fprintf(stderr, "No kernel error detected\n");

    // TODO: Now that kernel calls have finished, copy the output signal
    //       back from the GPU to host memory. (We store this channel's result
    //       in out_data on the host.)
    cudaMemcpy(out_data, gpu_out_data, n_frames*sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: Now that we have finished our computations on the GPU, free the
    //       GPU resources.
    cudaFree(gpu_raw_data);
    cudaFree(gpu_blur_v);
    cudaFree(gpu_out_data);

    // Stop the recording timer and return the computation time
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time_milli, start_gpu, stop_gpu);
    return time_milli;
}
