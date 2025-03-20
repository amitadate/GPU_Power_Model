#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "opt_2dhisto.h"
#include "ref_2dhisto.h"
#include "util.h"
#include <algorithm>
#include <cutil.h>
#include <math.h>

#define PAD_WIDTH ((INPUT_WIDTH + 128) & 0xFFFFFF80)
#define INPUT_SIZE (INPUT_WIDTH*INPUT_HEIGHT)
#define TILE_SIZE (32)


__global__ void Baseline(uint32_t* dInput, uint32_t* dBins)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        uint32_t index = dInput[col + (row * PAD_WIDTH)];
        atomicAdd(&dBins[index], 1);
    }
}

__global__ void ParallelStrategyOne(uint32_t* dInput, uint32_t* dBins, size_t rectangle_width, size_t rectangle_height) {
    int col = blockIdx.x * rectangle_width;
    int row = blockIdx.y * rectangle_height; 
   
    uint32_t privateHist[HISTO_HEIGHT * HISTO_WIDTH] = {0};
    for (size_t i = 0; i < rectangle_height; i++){
        for (size_t j = 0; j < rectangle_width; j++){
            size_t segmentCol = col + j;
            size_t segmentRow = row + i; 
            if (segmentCol < INPUT_WIDTH && segmentRow < INPUT_HEIGHT){
                uint32_t index = dInput[segmentCol + segmentRow * PAD_WIDTH];
                if (index < HISTO_HEIGHT * HISTO_WIDTH){
                    privateHist[index]++;
                }              
            }           
        }
    }

    for (uint32_t idx = 0; idx < HISTO_HEIGHT * HISTO_WIDTH; idx++) {
        atomicAdd(&(dBins[idx]), privateHist[idx]); // Update global histogram safely
    }
}

__global__ void ParallelStrategyTwo(uint32_t* dInput, uint32_t* dBins)
{
    __shared__ uint32_t shared_hist[HISTO_HEIGHT * HISTO_WIDTH];

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // initialize the shared memory bins
    for (int i = threadIdx.x; i < HISTO_HEIGHT * HISTO_WIDTH;
         i = i + blockDim.x) {
        shared_hist[threadIdx.x] = 0;
    }
    __syncthreads();  // initialization synced

    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        uint32_t index = dInput[col + (row * PAD_WIDTH)];
        atomicAdd(&dBins[index], 1);
    }
    __syncthreads();  // final synced (shared memory is now updated)

    // copy results back to global memory from shared memory
    for (int i = threadIdx.x; i < HISTO_HEIGHT * HISTO_WIDTH;
         i = i + blockDim.x) {
        atomicAdd(&dBins[i], shared_hist[i]);
    }
}

__global__ void ParallelStrategyThree(uint32_t* dInput, uint32_t* dBins)
{
    __shared__ uint32_t shared_hist[255];  
    __shared__ uint32_t input_tile[32 * 32];
    
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    for (int i = tid; i < 255; i += blockDim.x * blockDim.y) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Load input tile
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        input_tile[tid] = dInput[col + (row * PAD_WIDTH)];
    }
    __syncthreads();
    
    // Process pixel
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        uint32_t value = input_tile[tid];
        atomicAdd(&dBins[value], 1);  
    }
}

__global__ void ParallelStrategyFour(uint32_t* dInput, uint32_t* dBins)
{
    __shared__ uint32_t shared_hist[255];
    __shared__ uint32_t input_tile[33 * 32];  // Added padding to avoid bank conflicts
    
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    for (int i = tid; i < 255; i += blockDim.x * blockDim.y) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Load input tile with using padded indexing
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        input_tile[tid + (threadIdx.y)] = dInput[col + (row * PAD_WIDTH)];  // Add threadIdx.y for padding
    }
    __syncthreads();
    
    // Process pixel 
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        uint32_t value = input_tile[tid + (threadIdx.y)];  // Access with padding offset
        atomicAdd(&dBins[value], 1);
    }
}

__global__ void ParallelStrategyFive(uint32_t* dInput, uint32_t* dBins)
{
    __shared__ uint32_t shared_hist[255];
    __shared__ uint32_t input_tile[33 * 32];
    
    // Calculate global linear index for coalesced access
    int blockOffset = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * PAD_WIDTH;
    int threadOffset = threadIdx.x + threadIdx.y * PAD_WIDTH;
    int globalIdx = blockOffset + threadOffset;
    
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    for (int i = tid; i < 255; i += blockDim.x * blockDim.y) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Load input tile using coalesced access pattern
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        input_tile[tid + (threadIdx.y)] = dInput[globalIdx];
    }
    __syncthreads();
    
    // Process pixel
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        uint32_t value = input_tile[tid + (threadIdx.y)];
        atomicAdd(&dBins[value], 1);
    }
}

__global__ void ParallelStrategySix(uint32_t* dInput, uint32_t* dBins) 
{
    __shared__ uint32_t input_tile[49 * 48];  // TILE == 48 (TILE 64 FAILED), 48 is a good limit 
    
    // Calculate global linear index for coalesced access
    int blockOffset = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * PAD_WIDTH;
    int threadOffset = threadIdx.x + threadIdx.y * PAD_WIDTH;
    int globalIdx = blockOffset + threadOffset;
    
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Load input tile using coalesced access pattern
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        input_tile[tid + (threadIdx.y)] = dInput[globalIdx];
    }
    __syncthreads();
    
    // Process pixel - write directly to global memory
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        uint32_t value = input_tile[tid + (threadIdx.y)];
        atomicAdd(&dBins[value], 1);
    }
}

__global__ void ParallelStrategySeven(uint32_t* dInput, uint32_t* dBins)
{
    // Optimize shared memory layout - use linear indexing
    __shared__ uint32_t input_tile[1024];  // 32*32 without extra padding
    
    // Calculate indices
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Calculate global linear index for coalesced memory access
    int globalIdx = col + (row * PAD_WIDTH);
    
    // Load input tile - simplified indexing
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        input_tile[tid] = dInput[globalIdx];
    }
    __syncthreads();
    
    // Process pixel - same direct atomic update
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        uint32_t value = input_tile[tid];
        atomicAdd(&dBins[value], 1);
    }
}

__global__ void ParallelStrategyEight(uint32_t* dInput, uint32_t* dBins)
{
    // Original shared memory
    __shared__ uint32_t input_tile[1024];  // 32*32 without extra padding
    
    // Add local histogram bins
    __shared__ uint32_t local_histogram[256];

    // Calculate indices
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Calculate global linear index for coalesced memory access
    int globalIdx = col + (row * PAD_WIDTH);
    
    // Initialize local histogram to 0
    if(tid < 256) {
        local_histogram[tid] = 0;
    }
    __syncthreads();
    
    // Original load input tile
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        input_tile[tid] = dInput[globalIdx];
    }
    __syncthreads();
    
    // Original process pixel
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        uint32_t value = input_tile[tid];
        atomicAdd(&dBins[value], 1);
    }
}

__global__ void ParallelStrategyNine(uint32_t* dInput, uint32_t* dBins)
{
    // Original shared memory
    __shared__ uint32_t input_tile[1024];  // 32*32 without extra padding
    __shared__ uint32_t localBins[1024];    // One bin per value

    // Calculate indices
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int globalIdx = col + (row * PAD_WIDTH);

    // Initialize local bins
    if(tid < 1024) {
        localBins[tid] = 0;
    }
    __syncthreads();

    // Load and process data
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH) {
        uint32_t value = dInput[globalIdx];
        atomicAdd(&localBins[value], 1);
    }
    __syncthreads();

    // Reduction - have each thread handle its own bin
    if(tid < 1024) {
        uint32_t count = localBins[tid];
        if(count > 0) {
            atomicAdd(&dBins[tid], count);
        }
    }
}

int ceilDiv(int a, int b)
{
    return (a + b - 1) / b;
}

// New function that selects which strategy to run
void opt_2dhisto_strategy(uint32_t* dInput, uint32_t* dBins, int strategy)
{
    // Clear bins before running any strategy
    cudaMemset(dBins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));

    // Grid and block dimensions
    dim3 gridDims(ceilDiv(PAD_WIDTH, 32), ceilDiv(INPUT_HEIGHT, 32));
    dim3 blockDims(32, 32);
    size_t sharedMemSize = 255 * sizeof(uint32_t);
    
    // Variables needed for strategy 1
    size_t num_of_cuts = 1;
    size_t num_segments_width = ceilDiv((INPUT_WIDTH), num_of_cuts);
    size_t num_segments_height = ceilDiv((INPUT_HEIGHT), num_of_cuts);
    size_t rectangle_width = (INPUT_WIDTH / num_segments_width);
    size_t rectangle_height = (INPUT_HEIGHT / num_segments_height);
    dim3 strategy1BlockDims(1, 1);
    dim3 strategy1GridDims(num_segments_width, num_segments_height);

    switch (strategy) {
        case 0: // Baseline
            Baseline<<<gridDims, blockDims>>>(dInput, dBins);
            break;
        case 1:
            // Use the variables for Strategy 1 which were already defined
            ParallelStrategyOne<<<strategy1GridDims, strategy1BlockDims>>>(dInput, dBins, rectangle_width, rectangle_height);
            break;
        case 2:
            ParallelStrategyTwo<<<gridDims, blockDims, sharedMemSize>>>(dInput, dBins);
            break;
        case 3:
            ParallelStrategyThree<<<gridDims, blockDims, sharedMemSize>>>(dInput, dBins);
            break;
        case 4:
            ParallelStrategyFour<<<gridDims, blockDims, sharedMemSize>>>(dInput, dBins);
            break;
        case 5:
            ParallelStrategyFive<<<gridDims, blockDims, sharedMemSize>>>(dInput, dBins);
            break;
        case 6:
            ParallelStrategySix<<<gridDims, blockDims, sharedMemSize>>>(dInput, dBins);
            break;
        case 7:
            ParallelStrategySeven<<<gridDims, blockDims, sharedMemSize>>>(dInput, dBins);
            break;
        case 8:
            ParallelStrategyEight<<<gridDims, blockDims, sharedMemSize>>>(dInput, dBins);
            break;
        case 9:
            ParallelStrategyNine<<<gridDims, blockDims, sharedMemSize>>>(dInput, dBins);
            break;
        default:
            // Default to strategy 9 if an invalid strategy is specified
            ParallelStrategyNine<<<gridDims, blockDims, sharedMemSize>>>(dInput, dBins);
    }

    // Ensure kernel execution is complete
    cudaDeviceSynchronize();
}

// Original function now calls strategy 9 by default
void opt_2dhisto(uint32_t* dInput, uint32_t* dBins)
{
    // Just call strategy 9 (the one that was previously being used)
    opt_2dhisto_strategy(dInput, dBins, 9);
}

uint32_t* allocateAndCopyDeviceInput(uint32_t** hInput, size_t height,
                                     size_t width, size_t elementSize)
{
    size_t    dInputSize = height * PAD_WIDTH * sizeof(uint32_t);
    uint32_t* dInput;

    cudaMalloc((void**) &dInput, height * PAD_WIDTH * sizeof(uint32_t));

    // correct indexing of hInput to dInput
    for (int i = 0; i < INPUT_HEIGHT; i++) {
        int offset = i * PAD_WIDTH;
        cudaMemcpy(&dInput[offset], hInput[i], PAD_WIDTH * sizeof(uint32_t),
                   cudaMemcpyHostToDevice);
    }

    uint32_t* dInputVerif =
            (uint32_t*) malloc(height * PAD_WIDTH * sizeof(uint32_t));
    for (int i = 0; i < INPUT_HEIGHT; i++) {
        int offset = i * PAD_WIDTH;
        cudaMemcpy((void*) &dInputVerif[offset], &dInput[offset],
                   PAD_WIDTH * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    printf("device input size is %zu\n", dInputSize);

    return dInput;
}

uint32_t* allocateAndOutputDeviceBins()
{
    uint32_t* dBins;
    cudaMalloc((void**) &dBins, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
    // first memset for initialization of bins
    cudaMemset(dBins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
    return dBins;
}

void dev_to_host_bins(uint8_t* final_bins, uint32_t* dBins)
{
    uint32_t temp_bins[HISTO_HEIGHT * HISTO_WIDTH];
    cudaMemcpy(temp_bins, dBins, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    // iterate through entire histogram, and limit count to 255
    for (int i = 0; i < HISTO_HEIGHT * HISTO_WIDTH; i++) {
        if (temp_bins[i] > 255)
            final_bins[i] = 255;
        else
            final_bins[i] = temp_bins[i];
    }
}

void free_device_memory(uint32_t* dInput, uint32_t* dBins)
{
    cudaFree(dInput);
    cudaFree(dBins);
}
