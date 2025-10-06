#include <migraphx/common_api/test_interface.hpp>

#include <migraphx/program.hpp>

#include "test_common.hpp"
#include "source1.hpp"
#include <iostream>

#include <hip/hip_runtime.h>

// sandbox sample
int main()
{
    std::cout <<"test_sample main\n";
    test::test_interface* obj = create_test();
    obj->foo();
    delete obj;

    sample_common::sample_struct sc;
    sc.hello();

    sample_1::sample_struct s1;
    s1.hello();

    sample_common::sample_logger slog;
    slog.log();

    migraphx::program p;
    migraphx::shape s;

    int* pi = nullptr;
    bool b = hipMalloc(&pi, sizeof(int));
    if (b)
    {
        auto hipFreeStatus = hipFree(pi);
        if (hipFreeStatus != hipSuccess)
        {
            std::cout <<"error during hipFree\n";
        }
    }

    return 0;
}
