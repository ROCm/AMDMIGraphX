
#include <onnx_test.hpp>

TEST_CASE(reducel1_dyn_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        // a shape with 4 dynamic dimensions
        auto l0      = mm->add_parameter("x",
                                    migraphx::shape{migraphx::shape::float_type,
                                                         {{3, 3}, {3, 5}, {4, 6, {5}}, {5, 7, {6}}}});
        auto abs_ins = mm->add_instruction(migraphx::make_op("abs"), l0);
        auto sum_ins =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {-2}}}), abs_ins);
        auto sq_ins = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {-2}}}), sum_ins);
        mm->add_return({sq_ins});

        migraphx::onnx_options options;
        options.map_dyn_input_dims["x"] = {{3, 3}, {3, 5}, {4, 6, {5}}, {5, 7, {6}}};
        auto prog                       = migraphx::parse_onnx("reducel1_dyn_test.onnx", options);

        EXPECT(p == prog);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        // No axes given in the onnx file.  Parser should default to all axes.
        auto l0      = mm->add_parameter("x",
                                    migraphx::shape{migraphx::shape::float_type,
                                                         {{3, 3}, {3, 5}, {4, 6, {5}}, {5, 7, {6}}}});
        auto abs_ins = mm->add_instruction(migraphx::make_op("abs"), l0);
        auto sum_ins =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1, 2, 3}}}), abs_ins);
        auto sq_ins =
            mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1, 2, 3}}}), sum_ins);
        mm->add_return({sq_ins});

        migraphx::onnx_options options;
        options.map_dyn_input_dims["x"] = {{3, 3}, {3, 5}, {4, 6, {5}}, {5, 7, {6}}};
        auto prog = migraphx::parse_onnx("reducel1_dyn_noaxes_test.onnx", options);

        EXPECT(p == prog);
    }
}
