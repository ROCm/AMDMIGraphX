#include <migraphx/normalize_ops.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/op/common.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct normalize_test_op
{
    std::vector<int64_t> axes = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.axes, "axes"));
    }

    migraphx::value attributes() const
    {
        migraphx::value attr;
        attr["axes"] = migraphx::value::array{migraphx::op::op_normalize_attributes::clip_max,
                                              migraphx::op::op_normalize_attributes::clip_min};
        return {{"normalize_axes", attr}};
    }

    std::string name() const { return "normalize_ops_test::test_op"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs[0];
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return {output_shape};
    }
};

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::normalize_ops{}, migraphx::dead_code_elimination{}});
}

migraphx::program create_gather(int64_t axis)
{
    migraphx::program p;
    migraphx::shape sd{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape si{migraphx::shape::int64_type, {2, 3}};
    auto di = p.add_parameter("data", sd);
    auto ii = p.add_parameter("ind", si);
    auto r  = p.add_instruction(migraphx::make_op("gather", {{"axis", axis}}), di, ii);
    p.add_return({r});

    return p;
}

TEST_CASE(gather_test)
{

    auto p1 = create_gather(-3);
    auto p2 = create_gather(0);
    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(gather_test_1)
{
    auto p1 = create_gather(1);
    auto p2 = create_gather(1);
    run_pass(p1);

    EXPECT(p1 == p2);
}

migraphx::program create_reduce_mean(const std::vector<int64_t>& axes)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    auto si = p.add_parameter("data", s);
    auto r  = p.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), si);
    p.add_return({r});

    return p;
}

TEST_CASE(reduce_mean_test)
{
    migraphx::program p1 = create_reduce_mean({0, 1, -1});
    migraphx::program p2 = create_reduce_mean({0, 1, 3});
    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(reduce_mean_test_1)
{
    migraphx::program p1 = create_reduce_mean({0, 1, 2});
    migraphx::program p2 = create_reduce_mean({0, 1, 2});
    run_pass(p1);

    EXPECT(p1 == p2);
}

migraphx::program create_slice(const std::vector<int64_t>& axes,
                               const std::vector<int64_t>& starts,
                               const std::vector<int64_t>& ends)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    auto si = p.add_parameter("data", s);
    auto r  = p.add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}), si);
    p.add_return({r});

    return p;
}

TEST_CASE(slice_test)
{
    migraphx::program p1 = create_slice({0, 1, -1}, {-5, 1, -3}, {2, 2, 8});
    migraphx::program p2 = create_slice({0, 1, 3}, {0, 1, 2}, {2, 2, 5});
    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(slice_test_1)
{
    migraphx::program p1 = create_slice({0, 1, 3}, {0, 1, -3}, {1, 2, 5});
    migraphx::program p2 = create_slice({0, 1, 3}, {0, 1, 2}, {1, 2, 5});
    run_pass(p1);

    EXPECT(p1 == p2);
}

migraphx::program create_test_op(const std::vector<int64_t>& axes)
{
    migraphx::program p;
    migraphx::shape sd{migraphx::shape::float_type, {2, 3, 4}};
    auto di = p.add_parameter("data", sd);
    auto r  = p.add_instruction(normalize_test_op{axes}, di);
    p.add_return({r});

    return p;
}

TEST_CASE(test_op)
{
    std::vector<int64_t> axes1 = {-4, 5};
    auto p1                    = create_test_op(axes1);

    std::vector<int64_t> axes2 = {1, 2};
    auto p2                    = create_test_op(axes2);

    run_pass(p1);
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
