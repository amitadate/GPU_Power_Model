#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t* dInput,  uint32_t* dBins);

/* Include below the function headers of any other functions that you implement
 */

void run_single_kernel(uint32_t* dInput, uint32_t* dBins, int kernel_id);

uint32_t* allocateAndCopyDeviceInput(uint32_t** hInput, size_t height,
                                     size_t width, size_t elementSize);

void free_device_memory(uint32_t *input,  uint32_t* dBins);

uint32_t* allocateAndOutputDeviceBins();


void dev_to_host_bins(uint8_t* final_bins, uint32_t* dBins);

#endif
