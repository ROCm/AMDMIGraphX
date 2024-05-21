
#include <tf_test.hpp>


TEST_CASE(stridedslice_masks_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 10, 3, 3}});
    // add literals for starts, ends, and strides in tf (NHWC format)
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type, {4}},
                    std::vector<int>{0, 1, 1, 0});
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type, {4}},
                    std::vector<int>{0, 0, 0, 0});
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type, {4}},
                    std::vector<int>{1, 1, 1, 1});

    auto l1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op(
            "slice", {{"starts", {0, 1, 1, 0}}, {"ends", {1, 3, 3, 10}}, {"axes", {0, 1, 2, 3}}}),
        l1);
    auto l3 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), l2);
    mm->add_return({l3});
    auto prog = parse_tf("stridedslice_masks_test.pb", true);

    EXPECT(p == prog);
}


