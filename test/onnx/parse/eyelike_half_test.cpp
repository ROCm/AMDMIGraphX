
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(eyelike_half_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<std::size_t> input_lens{8, 8};
    const size_t k   = 0;
    auto num_rows    = input_lens.front();
    auto num_cols    = input_lens.back();
    auto input_type  = migraphx::shape::half_type;
    auto output_type = migraphx::shape::half_type;
    migraphx::shape s{input_type, input_lens};
    mm->add_parameter("T1", s);

    auto eyelike_mat = make_r_eyelike(num_rows, num_cols, k);
    mm->add_literal(migraphx::literal{migraphx::shape{output_type, input_lens}, eyelike_mat});

    auto prog = optimize_onnx("eyelike_half_test.onnx");
    EXPECT(p == prog);
}


