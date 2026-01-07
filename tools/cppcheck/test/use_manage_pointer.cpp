// Test for UseManagePointer check
// Note: This test file defines mock functions to avoid actual memory operations.
// The mock functions are defined before use to satisfy cppcheck's analysis.

// Forward declarations and mocks for HIP functions
// TODO: migraphx-UseManagePointer false positive - forward declarations trigger the check
typedef struct FILE FILE;
// cppcheck-suppress migraphx-UseManagePointer
int fclose(FILE*);
// cppcheck-suppress migraphx-UseManagePointer
void free(void*);
void* malloc(unsigned long);
// cppcheck-suppress migraphx-UseManagePointer
int hipFree(void*);
// cppcheck-suppress migraphx-UseManagePointer
int hipHostFree(void*);
// cppcheck-suppress migraphx-UseManagePointer
int hipStreamDestroy(void*);
// cppcheck-suppress migraphx-UseManagePointer
int hipEventDestroy(void*);
int hipMalloc(void**, unsigned long);

void test_fclose(FILE* file)
{
    // cppcheck-suppress migraphx-UseManagePointer
    fclose(file);
}

void test_free(void* ptr)
{
    // cppcheck-suppress migraphx-UseManagePointer
    free(ptr);
}

void test_hip_functions(void* gpu_ptr)
{
    // cppcheck-suppress migraphx-UseManagePointer
    hipFree(gpu_ptr);

    // cppcheck-suppress migraphx-UseManagePointer
    hipHostFree(gpu_ptr);

    // cppcheck-suppress migraphx-UseManagePointer
    hipStreamDestroy(gpu_ptr);

    // cppcheck-suppress migraphx-UseManagePointer
    hipEventDestroy(gpu_ptr);
}

void test_negative_cases()
{
    // Should not trigger: other functions
    int x    = 5;
    int* ptr = &x;
    (void)x; // Use variables to avoid warnings
    (void)ptr;

    // Should not trigger: allocation functions (not deallocation)
    hipMalloc(nullptr, 100);
}

// Mock function implementations
// TODO: migraphx-UseManagePointer false positive - function definitions trigger the check
// cppcheck-suppress migraphx-UseManagePointer
int fclose(FILE*) { return 0; }
// cppcheck-suppress migraphx-UseManagePointer
void free(void*) {}
void* malloc(unsigned long) { return nullptr; }
// cppcheck-suppress migraphx-UseManagePointer
int hipFree(void*) { return 0; }
// cppcheck-suppress migraphx-UseManagePointer
int hipHostFree(void*) { return 0; }
// cppcheck-suppress migraphx-UseManagePointer
int hipStreamDestroy(void*) { return 0; }
// cppcheck-suppress migraphx-UseManagePointer
int hipEventDestroy(void*) { return 0; }
int hipMalloc(void**, unsigned long) { return 0; }
