
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(mvn_rank_3_test)
{
    mvn_n_rank_test({0, 1}, {2, 2, 2}, optimize_onnx("mvn_rank_3_test.onnx"));
}
