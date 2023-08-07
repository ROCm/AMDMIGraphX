
#ifdef __HIP_DEVICE_COMPILE__
#error \
    "Device compilation not allowed for migraphx_gpu. Do not link with hip::device. Device code should go into migraphx_device or migraphx_kernels"
#endif
