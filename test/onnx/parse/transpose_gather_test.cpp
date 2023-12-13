
#include <onnx_test.hpp>


TEST_CASE(transpose_gather_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    auto make_contiguous = [&mm](migraphx::instruction_ref ins) {
        if(ins->get_shape().standard())
        {
            return ins;
        }

        return mm->add_instruction(migraphx::make_op("contiguous"), ins);
    };

    auto data =
        mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 5, 4, 6}});
    auto ind =
        mm->add_parameter("indices", migraphx::shape{migraphx::shape::int32_type, {2, 4, 3, 5}});
    auto tr_data =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), data);
    auto tr_ind =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), ind);
    int axis = 1;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}),
                        make_contiguous(tr_data),
                        make_contiguous(tr_ind));

    auto prog = optimize_onnx("transpose_gather_test.onnx");

    EXPECT(p.sort() == prog.sort());
}


