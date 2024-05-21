
#include <tf_test.hpp>
#include <tf_conv_utils.hpp>


TEST_CASE(conv_test)
{
    migraphx::program p = create_conv();
    auto prog           = optimize_tf("conv_test.pb", true);

    EXPECT(p == prog);
}


