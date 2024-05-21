
#include <tf_test.hpp>
#include <tf_conv_utils.hpp>


TEST_CASE(conv_nchw_test)
{
    migraphx::program p = create_conv();
    auto prog           = optimize_tf("conv_nchw_test.pb", false);

    EXPECT(p == prog);
}


