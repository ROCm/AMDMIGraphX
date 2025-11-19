// Test for UseManagePointer check
#include <cstdlib>

void test_positive_cases()
{
    // Should trigger: fclose usage
    FILE* file = (FILE*)malloc(sizeof(FILE));
    // cppcheck-suppress migraphx-UseManagePointer
    fclose(file);

    // Should trigger: free usage
    void* ptr = malloc(100);
    // cppcheck-suppress migraphx-UseManagePointer
    free(ptr);

    // Should trigger: HIP memory functions
    void* gpu_ptr = nullptr;
    // cppcheck-suppress migraphx-UseManagePointer
    hipFree(gpu_ptr);

    // cppcheck-suppress migraphx-UseManagePointer
    hipHostFree(gpu_ptr);

    // cppcheck-suppress migraphx-UseManagePointer
    hipStreamDestroy(nullptr);

    // cppcheck-suppress migraphx-UseManagePointer
    hipEventDestroy(nullptr);
}

void test_negative_cases()
{
    // Should not trigger: using smart pointers
    // std::unique_ptr<int> smart_ptr(new int(5));

    // Should not trigger: RAII wrappers
    // manage_ptr<FILE> file_ptr(fopen("test.txt", "r"), fclose);

    // Should not trigger: other functions
    int x    = 5;
    int* ptr = &x;
    (void)x; // Use variables to avoid warnings
    (void)ptr;

    // Should not trigger: allocation functions
    void* allocated     = malloc(100);
    void* hip_allocated = nullptr;
    hipMalloc(&hip_allocated, 100);

    // Clean up to avoid memory leaks
    free(allocated);
    hipFree(hip_allocated);
}

// Mock functions for compilation
typedef struct FILE FILE;
int fclose(FILE*) { return 0; }
void free(void*) {}
void* malloc(size_t) { return nullptr; }
int hipFree(void*) { return 0; }
int hipHostFree(void*) { return 0; }
int hipStreamDestroy(void*) { return 0; }
int hipEventDestroy(void*) { return 0; }
int hipMalloc(void**, size_t) { return 0; }
