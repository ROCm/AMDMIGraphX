
#include <tf_test.hpp>

TEST_CASE(pack_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2}});
    auto l2  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {2}});
    std::vector<migraphx::instruction_ref> args{l0, l1, l2};
    std::vector<migraphx::instruction_ref> unsqueezed_args;
    int64_t axis = 1;

    std::transform(
        args.begin(),
        args.end(),
        std::back_inserter(unsqueezed_args),
        [&](migraphx::instruction_ref arg) {
            return mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {axis}}}), arg);
        });
    mm->add_instruction(migraphx::make_op("concat", {{"axis", static_cast<int>(axis)}}),
                        unsqueezed_args);
    auto prog = optimize_tf("pack_test.pb", false);

    EXPECT(p == prog);
}


