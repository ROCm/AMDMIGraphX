
#include <onnx_test.hpp>


TEST_CASE(gather_elements_axis0_test)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto data    = mm->add_parameter("data", {migraphx::shape::float_type, {3, 4}});
    auto indices = mm->add_parameter("indices", {migraphx::shape::int32_type, {2, 3}});
    std::vector<int> ind_indices{0, 1, 2, 4, 5, 6};
    std::vector<int> ind_axis_indices{0, 0, 0, 1, 1, 1};
    migraphx::shape ind_s{migraphx::shape::int32_type, {2, 3}};
    auto l_data_indices =
        mm->add_literal(migraphx::literal{ind_s, ind_indices.begin(), ind_indices.end()});
    auto l_ind_axis_indices =
        mm->add_literal(migraphx::literal{ind_s, ind_axis_indices.begin(), ind_axis_indices.end()});
    auto l_stride = mm->add_literal(migraphx::literal{{migraphx::shape::int32_type, {1}}, {4}});

    auto rsp_data    = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
    auto lbst_stride = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", ind_s.lens()}}), l_stride);
    auto axis_delta = mm->add_instruction(migraphx::make_op("sub"), indices, l_ind_axis_indices);
    auto mul_delta  = mm->add_instruction(migraphx::make_op("mul"), axis_delta, lbst_stride);
    auto ind        = mm->add_instruction(migraphx::make_op("add"), l_data_indices, mul_delta);
    auto ret = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp_data, ind);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("gather_elements_axis0_test.onnx");

    EXPECT(p == prog);
}


