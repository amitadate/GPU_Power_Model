#ifndef OPT_KERNEL
#define OPT_KERNEL

// Main function that selects which strategy to run
void opt_2dhisto_strategy(uint32_t* dInput, uint32_t* dBins, int strategy);

// Original function (will be implemented to call strategy 9 by default)
void opt_2dhisto(uint32_t* dInput, uint32_t* dBins);

/* Include below the function headers of any other functions that you implement */

uint32_t* allocateAndCopyDeviceInput(uint32_t** hInput, size_t height,
                                     size_t width, size_t elementSize);

void free_device_memory(uint32_t *input, uint32_t* dBins);

uint32_t* allocateAndOutputDeviceBins();

void dev_to_host_bins(uint8_t* final_bins, uint32_t* dBins);

#endif
