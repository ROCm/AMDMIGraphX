#include "test_utils.hpp"
#include <iostream>

namespace test_utils
{
    test::MyInt foo()
    {
        std::cout << "Hello from test_utils::foo()" << std::endl;
        return 0;
    }
}
