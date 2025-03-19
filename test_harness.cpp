#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <stdio.h> //Needed for printf
#include <cuda_runtime.h> 
#include <string.h>
#include <cuda.h>
#include <cutil.h>

#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

#define SQRT_2    1.4142135623730950488
#define SPREAD_BOTTOM   (2)
#define SPREAD_TOP      (6)

#define NEXT(init_, spread_)\
    (init_ + (int)((drand48() - 0.5) * (drand48() - 0.5) * 4.0 * SQRT_2 * SQRT_2 * spread_));

#define CLAMP(value_, min_, max_)\
    if (value_ < 0)\
        value_ = (min_);\
    else if (value_ > (max_))\
        value_ = (max_);

// Generate another bin for the histogram.  The bins are created as a random walk ...
static uint32_t next_bin(uint32_t pix)
{
    const uint16_t bottom = pix & ((1<<HISTO_LOG)-1);
    const uint16_t top   = (uint16_t)(pix >> HISTO_LOG);

    int new_bottom = NEXT(bottom, SPREAD_BOTTOM)
    CLAMP(new_bottom, 0, HISTO_WIDTH-1)

    int new_top = NEXT(top, SPREAD_TOP)
    CLAMP(new_top, 0, HISTO_HEIGHT-1)

    const uint32_t result = (new_bottom | (new_top << HISTO_LOG)); 

    return result; 
}

// Return a 2D array of histogram bin-ids.  This function generates
// bin-ids with correlation characteristics similar to some actual images.
// The key point here is that the pixels (and thus the bin-ids) are *NOT*
// randomly distributed ... a given pixel tends to be similar to the
// pixels near it.
static uint32_t **generate_histogram_bins()
{
    uint32_t **input = (uint32_t**)alloc_2d(INPUT_HEIGHT, INPUT_WIDTH, sizeof(uint32_t));

    input[0][0] = HISTO_WIDTH/2 | ((HISTO_HEIGHT/2) << HISTO_LOG);
    for (int i = 1; i < INPUT_WIDTH; ++i)
        input[0][i] =  next_bin(input[0][i - 1]);
    for (int j = 1; j < INPUT_HEIGHT; ++j)
    {
        input[j][0] =  next_bin(input[j - 1][0]);
        for (int i = 1; i < INPUT_WIDTH; ++i)
            input[j][i] =  next_bin(input[j][i - 1]);
    }

    return input;
}

int main(int argc, char* argv[])
{
    /* Case of 0 arguments: Default seed is used */
    if (argc < 2){
        srand48(0);
    }
    /* Case of 1 argument: Seed is specified as first command line argument */ 
    else {
        int seed = atoi(argv[1]);
        srand48(seed);
    }

    uint8_t *gold_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));

    // Use kernel_bins for your final result
    uint8_t *kernel_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));

    // A 2D array of histogram bin-ids.  One can think of each of these bins-ids as
    // being associated with a pixel in a 2D image.
    uint32_t **input = generate_histogram_bins();

    TIME_IT("ref_2dhisto",
            1000,
            ref_2dhisto(input, INPUT_HEIGHT, INPUT_WIDTH, gold_bins);)

    /* Include your setup code below (temp variables, function calls, etc.) */
    uint32_t* dBins = allocateAndOutputDeviceBins();
    uint32_t* dInput = allocateAndCopyDeviceInput(input, INPUT_HEIGHT, INPUT_WIDTH, sizeof(uint32_t));
    /* End of setup code */

    // Check for command-line options
    bool run_individual = false;
    int specific_kernel = -1;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--individual") == 0) {
            run_individual = true;
        }
        else if (strcmp(argv[i], "--kernel") == 0 && i+1 < argc) {
            specific_kernel = atoi(argv[i+1]);
            i++; // Skip the next argument which is the kernel ID
        }
    }
    
    // Check if a specific kernel was requested
    if (specific_kernel >= 0 && specific_kernel <= 9) {
        // Run the specific kernel
        cudaMemset(dBins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
        
        const char* kernel_names[] = {
            "Baseline", "Strategy1", "Strategy2", "Strategy3", "Strategy4", 
            "Strategy5", "Strategy6", "Strategy7", "Strategy8", "Strategy9"
        };
        
        printf("Running %s kernel (ID: %d)...\n", kernel_names[specific_kernel], specific_kernel);
        
        // Time this specific kernel
        TIME_IT(kernel_names[specific_kernel],
                1,
                run_single_kernel(dInput, dBins, specific_kernel))
    }
    else if (run_individual) {
        /* Run each kernel strategy with separate timing */
        printf("KERNEL_TIMING: BEGIN\n");
        
        // Define kernel names
        const char* kernel_names[] = {
            "Baseline", "Strategy1", "Strategy2", "Strategy3", "Strategy4", 
            "Strategy5", "Strategy6", "Strategy7", "Strategy8", "Strategy9"
        };
        
        // Run each kernel with timing
        for (int kernel_id = 0; kernel_id <= 9; kernel_id++) {
            // Clear bins before each kernel
            cudaMemset(dBins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
            
            printf("KERNEL_TIMING: STRATEGY=%s\n", kernel_names[kernel_id]);
            
            // Time this specific kernel
            struct timeval start, end;
            gettimeofday(&start, NULL);
            
            run_single_kernel(dInput, dBins, kernel_id);
            
            cudaDeviceSynchronize();
            
            gettimeofday(&end, NULL);
            double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
            
            printf("KERNEL_TIMING: TIME=%.6f\n", time_sec);
            
            // Verify correctness
            uint8_t *temp_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));
            dev_to_host_bins(temp_bins, dBins);
            
            bool passed = true;
            for (int i = 0; i < HISTO_HEIGHT*HISTO_WIDTH; i++){
                if (gold_bins[i] != temp_bins[i]){
                    passed = false;
                    break;
                }
            }
            
            printf("KERNEL_TIMING: PASSED=%s\n", passed ? "true" : "false");
            
            free(temp_bins);
        }
        
        printf("KERNEL_TIMING: END\n");
        
        // For compatibility, also run all kernels together
        cudaMemset(dBins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
        
        TIME_IT("opt_2dhisto_all",
                1,
                opt_2dhisto(dInput, dBins))
    }
    else {
        /* This is the call you will use to time your parallel implementation */
        TIME_IT("opt_2dhisto",
                1000,
                opt_2dhisto(dInput, dBins))
    }

    /* Include your teardown code below (temporary variables, function calls, etc.) */
    dev_to_host_bins(kernel_bins, dBins);
    free_device_memory(dInput,dBins);
    /* End of teardown code */

    int passed=10;
    for (int i=0; i < HISTO_HEIGHT*HISTO_WIDTH; i++){
        if (gold_bins[i] != kernel_bins[i]){
            passed = 0;
            break;
        }
       //printf("\n true_count, output_count  %u %u", gold_bins[i], kernel_bins[i]);
    }
    (passed) ? printf("\n    Test PASSED\n") : printf("\n    Test FAILED\n");

    free(gold_bins);
    free(kernel_bins);
}
