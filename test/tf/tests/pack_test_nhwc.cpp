
#include <tf_test.hpp>

TEST_CASE(pack_test_nhwc)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
    auto lt0 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
    auto l1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
    auto lt1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l1);
    auto l2 = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
    auto lt2 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l2);
    std::vector<migraphx::instruction_ref> args{lt0, lt1, lt2};
    std::vector<migraphx::instruction_ref> unsqueezed_args;
    int64_t nchw_axis = 3;

    std::transform(args.begin(),
                   args.end(),
                   std::back_inserter(unsqueezed_args),
                   [&](migraphx::instruction_ref arg) {
                       return mm->add_instruction(
                           migraphx::make_op("unsqueeze", {{"axes", {nchw_axis}}}), arg);
                   });
    mm->add_instruction(migraphx::make_op("concat", {{"axis", static_cast<int>(nchw_axis)}}),
                        unsqueezed_args);
    auto prog = optimize_tf("pack_test_nhwc.pb", true);

    EXPECT(p == prog);
}


