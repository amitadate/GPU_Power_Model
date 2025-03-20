#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <stdio.h> //Needed for printf
#include <cuda_runtime.h> 

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

void run_histogram_strategy(uint32_t **input, uint8_t *gold_bins, int strategy_num) {
    printf("\n=== Running Histogram Strategy %d ===\n", strategy_num);
    
    // Create kernel_bins for each strategy's result
    uint8_t *kernel_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));
    
    // Allocate device memory
    uint32_t* dBins = allocateAndOutputDeviceBins();
    uint32_t* dInput = allocateAndCopyDeviceInput(input, INPUT_HEIGHT, INPUT_WIDTH, sizeof(uint32_t));
    
    // Time the strategy execution
    char strategy_name[50];
    sprintf(strategy_name, "opt_2dhisto_strategy_%d", strategy_num);
    
    TIME_IT(strategy_name, 1000, opt_2dhisto_strategy(dInput, dBins, strategy_num));
    
    // Get results back to host
    dev_to_host_bins(kernel_bins, dBins);
    
    // Verify results
    int passed = 1;
    for (int i = 0; i < HISTO_HEIGHT*HISTO_WIDTH; i++) {
        if (gold_bins[i] != kernel_bins[i]) {
            passed = 0;
            break;
        }
    }
    printf("    Strategy %d Test %s\n", strategy_num, (passed) ? "PASSED" : "FAILED");
    
    // Clean up
    free(kernel_bins);
    free_device_memory(dInput, dBins);
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
    
    printf("Starting histogram tests with all strategies\n");

    uint8_t *gold_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));

    // A 2D array of histogram bin-ids.  One can think of each of these bins-ids as
    // being associated with a pixel in a 2D image.
    uint32_t **input = generate_histogram_bins();

    // Run reference implementation to get gold standard results
    TIME_IT("ref_2dhisto", 1000, ref_2dhisto(input, INPUT_HEIGHT, INPUT_WIDTH, gold_bins));
    
    // Run all strategies (0 = baseline, 1-9 = different strategies)
    for (int strategy = 0; strategy <= 9; strategy++) {
        // Skip strategy 1 because it's too slow
        if (strategy == 1) continue;
        run_histogram_strategy(input, gold_bins, strategy);
    }
    
    printf("\nAll histogram tests completed.\n");
    free(gold_bins);

    return 0;
}
