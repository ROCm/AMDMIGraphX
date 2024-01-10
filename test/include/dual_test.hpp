
#ifdef __HIPCC_RTC__

#define DUAL_TEST_CASE() __device__ void tests()

__global__ void run() { tests(); }

#else
#include <test.hpp>

#define ROCM_DUAL_TEST_CASE() TEST_CASE(tests)


int main(int argc, const char* argv[]) { test::run(argc, argv); }
#endif
