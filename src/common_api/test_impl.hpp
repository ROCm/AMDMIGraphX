#ifndef TEST_IMPL_GUARD_H
#define TEST_IMPL_GUARD_H

#include <migraphx/common_api/test_interface.hpp>

namespace test
{
    class test_impl : public test_interface
    {
    public:
        void foo() override;
    };
}

#endif // TEST_IMPL_GUARD_H
