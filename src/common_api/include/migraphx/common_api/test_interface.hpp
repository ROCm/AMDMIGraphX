#ifndef TEST_GUARD_H
#define TEST_GUARD_H

#include <hip/hip_runtime.h>

namespace test
{
    class test_interface
    {
    public:
        virtual ~test_interface() = default;
        virtual void foo() = 0;
    };

    class test_logger
    {
    public:
        virtual ~test_logger() = default;
        virtual void log() = 0;
    };

    using MyInt = int;
}

extern "C" test::test_interface* create_test();

#endif // TEST_GUARD_H
