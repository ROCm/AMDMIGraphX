
#include <tf_test.hpp>


TEST_CASE(noop_test)
{
    migraphx::program p;
    auto prog = optimize_tf("noop_test.pb", false);

    EXPECT(p == prog);
}


