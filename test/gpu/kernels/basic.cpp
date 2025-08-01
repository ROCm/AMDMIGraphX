#include <migraphx/kernels/test.hpp>

TEST_CASE(always_true) { EXPECT(true); }

TEST_CASE(compare)
{
    int x = 2;
    int y = 2;
    EXPECT(x == y);
}
