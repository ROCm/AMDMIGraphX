#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/serialize.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::simplify_reshapes{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(double_contig)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c1 = mm->add_instruction(migraphx::make_op("contiguous"), t1);
    auto c2 = mm->add_instruction(migraphx::make_op("contiguous"), c1);
    mm->add_return({c2});
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 4);
    auto result = p.eval({}).back();
    EXPECT(result != get_2x2());
}

TEST_CASE(double_transpose)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto t2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), t1);
    mm->add_return({t2});
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == get_2x2());
}

TEST_CASE(double_transpose_contig)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c1 = mm->add_instruction(migraphx::make_op("contiguous"), t1);
    auto t2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), c1);
    auto c2 = mm->add_instruction(migraphx::make_op("contiguous"), t2);
    mm->add_return({c2});
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == get_2x2());
}

TEST_CASE(single_transpose)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    mm->add_return({t1});
    EXPECT(not mm->get_output_shapes().back().standard());
    EXPECT(mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(not mm->get_output_shapes().back().standard());
    EXPECT(mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 3);
    auto result = p.eval({}).back();
    EXPECT(result != get_2x2());
}

TEST_CASE(double_transpose_sin_pass)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), t1);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    // TODO: Fix this
    // EXPECT(std::distance(mm->begin(), mm->end()) == 1);
    auto result = p.eval({}).back();
    EXPECT(result == get_2x2());
}

TEST_CASE(single_transpose_sin_pass)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l = mm->add_literal(get_2x2());
    mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    EXPECT(not mm->get_output_shapes().back().standard());
    EXPECT(mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(not mm->get_output_shapes().back().standard());
    EXPECT(mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result != get_2x2());
}

TEST_CASE(reshape_transpose)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 112, 56, 56}};
    auto x  = m.add_parameter("x", s);
    auto r1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 28, 56, 56}}}), x);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3, 4}}}), r1);
    auto ct = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto r2 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 112, 56, 56}}}), ct);
    m.add_return({r2});
    EXPECT(m.get_output_shapes().back() == s);
    auto n = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == s);
    EXPECT(std::distance(m.begin(), m.end()) == n);
}

TEST_CASE(transpose_contiguous)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {4, 4}};
    auto x  = m.add_parameter("x", s);
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
    auto c1 = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_return({c1});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n);
}

TEST_CASE(transpose_double_contiguous)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {4, 4}};
    auto x  = m.add_parameter("x", s);
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
    auto c1 = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto c2 = m.add_instruction(migraphx::make_op("contiguous"), c1);
    m.add_return({c2});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
    EXPECT(m.has_instruction(t));
}

TEST_CASE(transpose_partial1)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = m.add_parameter("x", s);
    auto t1 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), x);
    auto t2 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t1);
    m.add_return({t2});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(transpose_partial2)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = m.add_parameter("x", s);
    auto t1 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), x);
    auto t2 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t1);
    auto t3 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), t2);
    m.add_return({t3});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
}

TEST_CASE(transpose_partial3)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = m.add_parameter("x", s);
    auto t1 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), x);
    auto t2 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t1);
    auto t3 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), t2);
    auto t4 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), t3);
    m.add_return({t4});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 3);
}

TEST_CASE(nop_transpose1)
{
    migraphx::module m;

    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x = m.add_parameter("x", s);
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), x);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(nop_transpose2)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = m.add_parameter("x", s);
    auto t1 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), x);
    auto t2 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), t1);
    auto t3 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), t2);
    auto t4 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), t3);
    m.add_instruction(pass_op{}, t4);
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 4);
}

TEST_CASE(nop_transpose3)
{
    migraphx::module m;

    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x      = m.add_parameter("x", s);
    auto y      = m.add_parameter("y", s);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), x, y);
    auto t1 =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), concat);
    auto t2 =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), t1);
    m.add_return({t2});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(nop_convert)
{
    migraphx::module m;

    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x = m.add_parameter("x", s);
    auto t = m.add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        x);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(concat_transpose1)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x  = m.add_parameter("x", s);
    auto y  = m.add_parameter("y", s);
    auto xt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), x);
    auto yt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), y);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), xt, yt);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), concat);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 3);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
    EXPECT(migraphx::any_cast<migraphx::op::concat>(new_concat->get_operator()).axis == 3);
}

TEST_CASE(concat_transpose2)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x  = m.add_parameter("x", s);
    auto y  = m.add_parameter("y", s);
    auto xt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
    auto yt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), y);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", -1}}), xt, yt);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), concat);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
    EXPECT(migraphx::any_cast<migraphx::op::concat>(new_concat->get_operator()).axis == 1);
}

TEST_CASE(concat_transpose3)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x  = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}});
    auto y  = m.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {1, 5, 3, 4}});
    auto xt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
    auto yt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), y);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), xt, yt);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), concat);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
    EXPECT(migraphx::any_cast<migraphx::op::concat>(new_concat->get_operator()).axis == 1);
}

TEST_CASE(concat_transpose4)
{
    migraphx::module m;
    auto sx = migraphx::shape{migraphx::shape::float_type, {1, 1, 12, 64}};
    auto sy = migraphx::shape{migraphx::shape::float_type, {1, 12, 1, 64}};
    auto x  = m.add_parameter("x", sx);
    auto y  = m.add_parameter("y", sy);
    auto xt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
    auto yt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), y);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), xt, yt);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), concat);
    m.add_return({t});

    migraphx::module m1 = m;
    run_pass(m);

    EXPECT(m1 == m);
}

TEST_CASE(nested_concat)
{
    migraphx::module m;

    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x       = m.add_parameter("x", s);
    auto y       = m.add_parameter("y", s);
    auto concat1 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
    auto concat2 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), y, x);
    auto concat3 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), concat1, concat2);
    m.add_return({concat3});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    EXPECT(std::count_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; }) == 1);
}

TEST_CASE(nested_concat_partial)
{
    migraphx::module m;

    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);
    auto l = m.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1, 4, 3, 4}}));
    auto concat1 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
    auto concat2 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), y, x);
    auto concat3 =
        m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), concat1, concat2, l);
    m.add_return({concat3});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    EXPECT(std::count_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; }) == 1);
}

TEST_CASE(multibroadcast_simplify)
{
    migraphx::module m;

    std::vector<size_t> s_lens{1, 2, 3, 4};
    auto s = migraphx::shape{migraphx::shape::float_type, s_lens};
    auto x = m.add_parameter("x", s);
    auto y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s_lens}}), x);
    m.add_instruction(migraphx::make_op("mul"), y, y);
    auto n = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(double_slice1)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {256}}}), x);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {64}}}), slice1);
        m1.add_return({slice2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {64}}, {"ends", {96}}}), x);
        m2.add_return({slice});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(double_slice2)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {128}}}), x);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {32}}}), slice1);
        m1.add_return({slice2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {64}}}), x);
        m2.add_return({slice});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(double_slice_multi_axes)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::int32_type, {256, 128}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {128}}}), x);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {32}}}), slice1);
        m1.add_return({slice2});
    }
    run_pass(m1);

    migraphx::module m2;

    {
        auto x     = m2.add_parameter("x", {migraphx::shape::int32_type, {256, 128}});
        auto slice = m2.add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {0, 1}}, {"starts", {32, 0}}, {"ends", {128, 32}}}),
            x);
        m2.add_return({slice});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(optimize_resize)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);

        migraphx::shape si{migraphx::shape::int32_type, {1, 2, 4, 6}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3,
                                3, 3, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 0, 0,
                                0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), gr);
        m.add_return({r});

        return m;
    };

    auto m1 = create_resize_module();
    run_pass(m1);

    auto create_optimized_module = [&] {
        migraphx::module m;
        auto inx                  = m.add_parameter("X", sx);
        std::vector<int64_t> dims = {1, 1, 2, 1, 2, 1};
        auto rspx = m.add_instruction(migraphx::make_op("reshape", {{"dims", dims}}), inx);
        std::vector<int64_t> mb_dims = {1, 2, 2, 2, 2, 3};
        auto mbx =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_dims}}), rspx);
        auto std_mb                    = m.add_instruction(migraphx::make_op("contiguous"), mbx);
        std::vector<int64_t> orig_dims = {1, 2, 4, 6};
        auto rmb = m.add_instruction(migraphx::make_op("reshape", {{"dims", orig_dims}}), std_mb);
        auto r   = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), rmb);
        m.add_return({r});

        return m;
    };

    EXPECT(m1 == create_optimized_module());
}

TEST_CASE(optimize_resize_ind_not_apply)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);

        migraphx::shape si{migraphx::shape::int32_type, {1, 2, 4, 6}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 2, 2, 2, 3,
                                3, 3, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 0, 0,
                                0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), gr);
        m.add_return({r});

        return m;
    };

    auto m1 = create_resize_module();
    run_pass(m1);
    EXPECT(m1 == create_resize_module());
}

TEST_CASE(optimize_resize_rsp_dim_1)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 3, 2}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2}}}), inx);
        auto r    = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        m.add_return({r});

        return m;
    };

    auto m = create_resize_module();
    run_pass(m);
    EXPECT(m == create_resize_module());
}

TEST_CASE(optimize_resize_ndims_unequal)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::shape sy{migraphx::shape::float_type, {1, 1, 4, 3, 2}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 3, 2}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("sub"), iny, gr);
        m.add_return({r});

        return m;
    };

    auto m = create_resize_module();
    run_pass(m);
    EXPECT(m == create_resize_module());
}

TEST_CASE(optimize_resize_ind_non_brcst)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
    migraphx::shape sy{migraphx::shape::float_type, {1, 1, 4, 6}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {6}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("sub"), iny, gr);
        m.add_return({r});

        return m;
    };

    auto m = create_resize_module();
    run_pass(m);
    EXPECT(m == create_resize_module());
}

TEST_CASE(optimize_resize_ind_non_const)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
    migraphx::shape sy{migraphx::shape::float_type, {1, 1, 4, 6}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
        auto li   = m.add_parameter("ind", si);
        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {6}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("sub"), iny, gr);
        m.add_return({r});

        return m;
    };

    auto m = create_resize_module();
    run_pass(m);
    EXPECT(m == create_resize_module());
}

TEST_CASE(optimize_where_true)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};
    auto create_where_module = [&](bool cond) {
        migraphx::module m;
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(si.elements(), static_cast<char>(cond));
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto return_xy = [&](bool cond) {
        migraphx::module m;
        auto x = m.add_parameter("X", s);
        auto y = m.add_parameter("Y", s);
        cond ? m.add_return({x}) : m.add_return({y});
        return m;
    };

    auto m = create_where_module(true);
    run_pass(m);
    EXPECT(m == return_xy(true));

    auto m1 = create_where_module(false);
    run_pass(m1);
    EXPECT(m1 == return_xy(false));
}

TEST_CASE(where_different_cond_values)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata = {1, 1, 0, 1, 0, 1};
        auto li                 = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(where_axis_nonzero)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(6, 1);
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(where_three_concat_inputs)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(6, 1);
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny, inx);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {18}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(where_three_inputs_diff_shapes)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 1, 3, 2}};
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(6, 1);
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {18}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(where_three_lens_diff)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
        migraphx::shape sy{migraphx::shape::float_type, {1, 1, 3, 2}};
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 6}};
        std::vector<char> idata(6, 1);
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(reshape_cont)
{
    auto create_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 2, 6}};

        auto inx = m.add_parameter("x", sx);
        auto iny = m.add_parameter("y", sy);
        auto mb_inx =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}), inx);
        auto std_inx = m.add_instruction(migraphx::make_op("contiguous"), mb_inx);
        auto rsp =
            m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 6}}}), std_inx);
        auto r = m.add_instruction(migraphx::make_op("add"), rsp, iny);
        m.add_return({r});

        return m;
    };

    auto m1 = create_module();
    run_pass(m1);

    auto create_opt_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 2, 6}};

        auto inx = m.add_parameter("x", sx);
        auto iny = m.add_parameter("y", sy);
        auto mb_inx =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}), inx);
        auto rsp_iny = m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4, 6}}}), iny);
        auto sum     = m.add_instruction(migraphx::make_op("add"), mb_inx, rsp_iny);
        auto r = m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 6}}}), sum);
        m.add_return({r});

        return m;
    };

    EXPECT(m1 == create_opt_module());
}

TEST_CASE(reshape_input_non_std)
{
    auto create_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 6, 2, 2}};

        auto inx = m.add_parameter("x", sx);
        auto iny = m.add_parameter("y", sy);
        auto mb_inx =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}), inx);
        auto std_inx = m.add_instruction(migraphx::make_op("contiguous"), mb_inx);
        auto rsp =
            m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 6}}}), std_inx);
        auto ty =
            m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), iny);
        auto r = m.add_instruction(migraphx::make_op("add"), rsp, ty);
        m.add_return({r});

        return m;
    };

    auto m1 = create_module();
    run_pass(m1);

    EXPECT(m1 == create_module());
}

TEST_CASE(reshape_cont_nonpw)
{
    auto create_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 2, 6}};

        auto inx = m.add_parameter("x", sx);
        auto iny = m.add_parameter("y", sy);
        auto mb_inx =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}), inx);
        auto std_inx = m.add_instruction(migraphx::make_op("contiguous"), mb_inx);
        auto rsp =
            m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 6}}}), std_inx);
        auto r = m.add_instruction(migraphx::make_op("convolution"), rsp, iny);
        m.add_return({r});

        return m;
    };

    auto m1 = create_module();
    run_pass(m1);

    EXPECT(m1 == create_module());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
