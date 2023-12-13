
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(mvn_default_axes_test)
{
    mvn_n_rank_test({0, 2, 3}, {2, 2, 2, 2}, optimize_onnx("mvn_default_axes_test.onnx"));
}


