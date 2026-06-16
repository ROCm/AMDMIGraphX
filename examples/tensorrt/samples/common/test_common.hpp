#ifndef TEST_COMMON_GUARD_H
#define TEST_COMMON_GUARD_H 

#include <migraphx/common_api/test_interface.hpp>

namespace sample_common
{
    struct sample_struct
    {
        void hello();
    };

    // implementing something from the interface
    class sample_logger : public test::test_logger
    {
    public:
        void log() override;
    };

} // namespace sample_common

#endif // TEST_COMMON_GUARD_H
