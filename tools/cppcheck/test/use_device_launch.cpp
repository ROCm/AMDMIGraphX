// Test for UseDeviceLaunch check

void test_positive_cases()
{
    // Should trigger: hipLaunchKernelGGL usage
    int gridSize  = 1;
    int blockSize = 256;

    // cppcheck-suppress migraphx-UseDeviceLaunch
    hipLaunchKernelGGL(kernel, gridSize, blockSize, 0, 0, args);

    // Should trigger: another hipLaunchKernelGGL call
    // cppcheck-suppress migraphx-UseDeviceLaunch
    hipLaunchKernelGGL(another_kernel, dim3(1), dim3(256), 0, nullptr);
}

void test_negative_cases()
{
    // Should not trigger: using device::launch instead
    // device::launch(kernel, gridSize, blockSize, args);

    // Should not trigger: other HIP functions
    hipMalloc(nullptr, 1024);
    hipMemcpy(nullptr, nullptr, 1024, hipMemcpyDeviceToHost);

    // Should not trigger: regular function calls
    printf("Hello world\n");

    // Should not trigger: custom functions with similar names
    myLaunchKernel(args);
}

// Mock functions and types for compilation
typedef int dim3;
typedef int hipMemcpyKind;
const hipMemcpyKind hipMemcpyDeviceToHost = 0;

void kernel() {}
void another_kernel() {}
int hipMalloc(void**, size_t) { return 0; }
int hipMemcpy(void*, const void*, size_t, hipMemcpyKind) { return 0; }
// TODO: migraphx-UseDeviceLaunch false positive - function definition triggers the check
// cppcheck-suppress migraphx-UseDeviceLaunch
int hipLaunchKernelGGL(void*, int, int, int, int, ...) { return 0; }
void myLaunchKernel(int) {}
int printf(const char*, ...) { return 0; }
