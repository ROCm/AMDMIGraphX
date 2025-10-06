#include "hip/hip_runtime.h"
#include "test_common.hpp"
#include <iostream>
#include "utils/test_utils.hpp"
#include "test_common2.hpp"

namespace sample_common
{
    void sample_struct::hello()
    {
        std::cout << "Hello from sample_common::sample_struct::hello()\n";
    }

    void sample_logger::log()
    {
        std::cout << "Hello from sample_common::sample_logger::log()\n";

        test_utils::foo();

        foo1();
    }
} // namespace sample_common
