#include <migraphx/program.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/pad_calc.hpp>
#include "test.hpp"


TEST_CASE(pad_calc_test_no_pad)
{

    std::vector<int64_t> golden_pads{0, 0, 0, 0};
    std::vector<int64_t> pads{0, 0, 0, 0};
    // 1x1 filter size
    migraphx::calculate_padding(0, pads, 16, 1, 1, 1);
    migraphx::calculate_padding(1, pads, 16, 1, 1, 1);
    EXPECT(pads == golden_pads);
}

TEST_CASE(pad_calc_test_pad_by_1)
{
    std::vector<int64_t> golden_pads{1, 1, 1, 1};
    std::vector<int64_t> pads{0, 0, 0, 0};
    // 3x3 filter size
    migraphx::calculate_padding(0, pads, 16, 1, 1, 3);
    migraphx::calculate_padding(1, pads, 16, 1, 1, 3);
    EXPECT(pads == golden_pads);
}

TEST_CASE(pad_calc_test_pad_by_1_asym_2x2_filter)
{
    std::vector<int64_t> golden_pads{0, 0, 1, 1};
    std::vector<int64_t> pads{0, 0, 0, 0};
    // 2x2 filter size
    migraphx::calculate_padding(0, pads, 16, 1, 1, 2);
    migraphx::calculate_padding(1, pads, 16, 1, 1, 2);
    EXPECT(pads == golden_pads);
}

TEST_CASE(pad_calc_test_pad_by_2)
{
    std::vector<int64_t> golden_pads{2, 2, 2, 2};
    std::vector<int64_t> pads{0, 0, 0, 0};
    // 5x5 filter size
    migraphx::calculate_padding(0, pads, 16, 1, 1, 5);
    migraphx::calculate_padding(1, pads, 16, 1, 1, 5);
    EXPECT(pads == golden_pads);
}

TEST_CASE(pad_calc_test_pad_by_1_asym_stride_2)
{
    std::vector<int64_t> golden_pads{0, 0, 1, 1};
    std::vector<int64_t> pads{0, 0, 0, 0};
    // 3x3 filter size
    migraphx::calculate_padding(0, pads, 16, 2, 1, 3);
    migraphx::calculate_padding(1, pads, 16, 2, 1, 3);
    EXPECT(pads == golden_pads);
}

TEST_CASE(pad_calc_test_dilation_2)
{
    std::vector<int64_t> golden_pads{2, 2, 2, 2};
    std::vector<int64_t> pads{0, 0, 0, 0};
    // 3x3 filter size
    migraphx::calculate_padding(0, pads, 16, 1, 2, 3);
    migraphx::calculate_padding(1, pads, 16, 1, 2, 3);
    EXPECT(pads == golden_pads);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
