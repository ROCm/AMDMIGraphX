
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(group_norm_5d_test)
{
    migraphx::program p = make_group_norm({3, 3, 3, 3, 3},
                                          {1},
                                          {1},
                                          {3, 1, 3, 3, 3, 3},
                                          {2, 3, 4, 5},
                                          1e-5f,
                                          migraphx::shape::float_type);
    auto prog           = optimize_onnx("group_norm_5d_test.onnx");
    EXPECT(p == prog);
}


