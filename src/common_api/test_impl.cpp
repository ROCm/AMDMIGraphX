#include "test_impl.hpp"
#include <hip/hip_runtime.h>
#include <iostream>

namespace test
{
    void test_impl::foo() 
    { 
        std::cout << "Hello from test_impl::foo()\n";

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
    }
} // namespace test

test::test_interface* create_test() 
{ 
    return new test::test_impl(); 
}

