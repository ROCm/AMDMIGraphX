#include <migraphx/auto_contiguous.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m) { migraphx::run_passes(m, {migraphx::auto_contiguous{}}); }

// TODO: Add this test case
void literal_broadcast()
{
    migraphx::module m;

    m.add_literal(get_2_broadcasted());
    EXPECT(not m.get_output_shapes().back().standard());
    EXPECT(m.get_output_shapes().back().broadcasted());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().broadcasted());
}

TEST_CASE(literal_transpose)
{
    migraphx::module m;

    m.add_literal(get_2x2_transposed());
    EXPECT(not m.get_output_shapes().back().standard());
    EXPECT(m.get_output_shapes().back().transposed());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().transposed());
}

TEST_CASE(after_literal_transpose)
{
    migraphx::module m;

    auto l = m.add_literal(get_2x2());
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().transposed());
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    m.add_instruction(pass_op{}, t);
    EXPECT(not m.get_output_shapes().back().standard());
    EXPECT(m.get_output_shapes().back().transposed());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().transposed());
}

TEST_CASE(after_literal_broadcast)
{
    migraphx::module m;

    auto l1 = m.add_literal(get_2x2());
    auto l2 = m.add_literal(get_2());
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().broadcasted());
    auto b = m.add_instruction(
        migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", l1->get_shape().lens()}}), l2);
    m.add_instruction(pass_op{}, b);
    EXPECT(not m.get_output_shapes().back().standard());
    EXPECT(m.get_output_shapes().back().broadcasted());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().broadcasted());
}

TEST_CASE(after_param_transpose)
{
    migraphx::module m;

    auto l = m.add_parameter("2x2", {migraphx::shape::float_type, {2, 2}});
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().transposed());
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    m.add_instruction(pass_op{}, t);
    EXPECT(not m.get_output_shapes().back().standard());
    EXPECT(m.get_output_shapes().back().transposed());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().transposed());
}

TEST_CASE(after_param_broadcast)
{
    migraphx::module m;

    auto l1 = m.add_parameter("2x2", {migraphx::shape::float_type, {2, 2}});
    auto l2 = m.add_parameter("2", {migraphx::shape::float_type, {2}});
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().broadcasted());
    auto b = m.add_instruction(
        migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", l1->get_shape().lens()}}), l2);
    m.add_instruction(pass_op{}, b);
    EXPECT(not m.get_output_shapes().back().standard());
    EXPECT(m.get_output_shapes().back().broadcasted());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().broadcasted());
}

TEST_CASE(two_transpose_gather)
{
    migraphx::module m1;
    {
        auto data = m1.add_parameter("2x2", {migraphx::shape::float_type, {2, 3, 4, 5}});
        auto ind  = m1.add_parameter("ind", {migraphx::shape::float_type, {2, 3}});
        auto td   = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), data);
        auto sd = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), td);
        auto bd =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), sd);
        auto r = m1.add_instruction(migraphx::make_op("gather", {{"axis", 2}}), bd, ind);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto data = m2.add_parameter("2x2", {migraphx::shape::float_type, {2, 3, 4, 5}});
        auto ind  = m2.add_parameter("ind", {migraphx::shape::float_type, {2, 3}});
        auto td   = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), data);
        auto ctd = m2.add_instruction(migraphx::make_op("contiguous"), td);
        auto sd  = m2.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), ctd);
        auto bd =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), sd);
        auto cbd = m2.add_instruction(migraphx::make_op("contiguous"), bd);
        auto r   = m2.add_instruction(migraphx::make_op("gather", {{"axis", 2}}), cbd, ind);
        m2.add_return({r});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(standard_reshape)
{
    migraphx::module m1;
    {
        auto data = m1.add_parameter("2x2", {migraphx::shape::float_type, {2, 3, 4, 5}});
        auto add  = m1.add_instruction(migraphx::make_op("add"), data, data);
        auto r = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1, 12, 5}}}), add);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto data = m2.add_parameter("2x2", {migraphx::shape::float_type, {2, 3, 4, 5}});
        auto add  = m2.add_instruction(migraphx::make_op("add"), data, data);
        auto ca   = m2.add_instruction(migraphx::make_op("contiguous"), add);
        auto r    = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1, 12, 5}}}), ca);
        m2.add_return({r});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(dead_instruction)
{
    migraphx::module m1;
    {
        auto data = m1.add_parameter("2x2", {migraphx::shape::float_type, {2, 3, 4, 5}});
        m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1, 3}}}), data);
        auto r = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1, 3}}}),
                                    data);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto data = m2.add_parameter("2x2", {migraphx::shape::float_type, {2, 3, 4, 5}});
        m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1, 3}}}), data);
        auto r = m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1, 3}}}),
                                    data);
        auto cr = m2.add_instruction(migraphx::make_op("contiguous"), r);
        m2.add_return({cr});
    }

    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
