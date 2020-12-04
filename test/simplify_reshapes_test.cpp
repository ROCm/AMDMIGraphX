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

void run_pass(migraphx::program& p)
{
    auto* mm = p.get_main_module();
    migraphx::run_passes(*mm, {migraphx::simplify_reshapes{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(double_contig)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    auto t1  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c1  = mm->add_instruction(migraphx::make_op("contiguous"), t1);
    auto c2  = mm->add_instruction(migraphx::make_op("contiguous"), c1);
    mm->add_return({c2});
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 4);
    auto result = p.eval({}).back();
    EXPECT(result != get_2x2());
}

TEST_CASE(double_transpose)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    auto t1  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto t2  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), t1);
    mm->add_return({t2});
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == get_2x2());
}

TEST_CASE(double_transpose_contig)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    auto t1  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c1  = mm->add_instruction(migraphx::make_op("contiguous"), t1);
    auto t2  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), c1);
    auto c2  = mm->add_instruction(migraphx::make_op("contiguous"), t2);
    mm->add_return({c2});
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == get_2x2());
}

TEST_CASE(single_transpose)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    auto t1  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    mm->add_return({t1});
    EXPECT(not p.get_output_shapes().back().standard());
    EXPECT(p.get_output_shapes().back().transposed());
    run_pass(p);
    EXPECT(not p.get_output_shapes().back().standard());
    EXPECT(p.get_output_shapes().back().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 3);
    auto result = p.eval({}).back();
    EXPECT(result != get_2x2());
}

TEST_CASE(double_transpose_sin_pass)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    auto t1  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), t1);
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
    // TODO: Fix this
    // EXPECT(std::distance(p.begin(), p.end()) == 1);
    auto result = p.eval({}).back();
    EXPECT(result == get_2x2());
}

TEST_CASE(single_transpose_sin_pass)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    EXPECT(not p.get_output_shapes().back().standard());
    EXPECT(p.get_output_shapes().back().transposed());
    run_pass(p);
    EXPECT(not p.get_output_shapes().back().standard());
    EXPECT(p.get_output_shapes().back().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result != get_2x2());
}

TEST_CASE(reshape_transpose)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {1, 112, 56, 56}};
    auto x   = mm->add_parameter("x", s);
    auto r1  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 28, 56, 56}}}), x);
    auto t   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 2, 1, 3, 4}}}), r1);
    auto ct  = mm->add_instruction(migraphx::make_op("contiguous"), t);
    auto r2  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 112, 56, 56}}}), ct);
    mm->add_return({r2});
    EXPECT(p.get_output_shapes().back() == s);
    auto n = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back() == s);
    EXPECT(std::distance(p.begin(), p.end()) == n);
}

TEST_CASE(transpose_contiguous)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {4, 4}};
    auto x   = mm->add_parameter("x", s);
    auto t   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), x);
    auto c1  = mm->add_instruction(migraphx::make_op("contiguous"), t);
    mm->add_return({c1});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n);
}

TEST_CASE(transpose_double_contiguous)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {4, 4}};
    auto x   = mm->add_parameter("x", s);
    auto t   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), x);
    auto c1  = mm->add_instruction(migraphx::make_op("contiguous"), t);
    auto c2  = mm->add_instruction(migraphx::make_op("contiguous"), c1);
    mm->add_return({c2});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 1);
    EXPECT(mm->has_instruction(t));
}

TEST_CASE(transpose_partial1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x   = mm->add_parameter("x", s);
    auto t1  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0, 2}}}), x);
    auto t2  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 2, 0}}}), t1);
    mm->add_return({t2});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 1);
}

TEST_CASE(transpose_partial2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x   = mm->add_parameter("x", s);
    auto t1  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0, 2}}}), x);
    auto t2  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 2, 0}}}), t1);
    auto t3  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0, 2}}}), t2);
    mm->add_return({t3});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 2);
}

TEST_CASE(transpose_partial3)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x   = mm->add_parameter("x", s);
    auto t1  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0, 2}}}), x);
    auto t2  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 2, 0}}}), t1);
    auto t3  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0, 2}}}), t2);
    auto t4  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0, 2}}}), t3);
    mm->add_return({t4});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 3);
}

TEST_CASE(nop_transpose1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x   = mm->add_parameter("x", s);
    auto t   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 2}}}), x);
    mm->add_return({t});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 1);
}

TEST_CASE(nop_transpose2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x   = mm->add_parameter("x", s);
    auto t1  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 2}}}), x);
    auto t2  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 2}}}), t1);
    auto t3  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 2}}}), t2);
    auto t4  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 2}}}), t3);
    mm->add_instruction(pass_op{}, t4);
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 4);
}

TEST_CASE(nop_transpose3)
{
    migraphx::program p;

    auto* mm    = p.get_main_module();
    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x      = mm->add_parameter("x", s);
    auto y      = mm->add_parameter("y", s);
    auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 3}}), x, y);
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 2, 3}}}), concat);
    auto t2 = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 3, 2}}}), t1);
    mm->add_return({t2});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 1);
}

TEST_CASE(nop_convert)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x   = mm->add_parameter("x", s);
    auto t   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx ::shape ::float_type)}}),
        x);
    mm->add_return({t});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 1);
}

TEST_CASE(concat_transpose1)
{
    migraphx::program p;

    auto* mm    = p.get_main_module();
    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x      = mm->add_parameter("x", s);
    auto y      = mm->add_parameter("y", s);
    auto xt     = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 3, 2}}}), x);
    auto yt     = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 3, 2}}}), y);
    auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), xt, yt);
    auto t = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 3, 2}}}), concat);
    mm->add_return({t});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(p.begin(), p.end()) == n - 3);
    auto new_concat =
        std::find_if(p.begin(), p.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != p.end()});
    EXPECT(migraphx::any_cast<migraphx::op::concat>(new_concat->get_operator()).axis == 3);
}

TEST_CASE(concat_transpose2)
{
    migraphx::program p;

    auto* mm    = p.get_main_module();
    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x      = mm->add_parameter("x", s);
    auto y      = mm->add_parameter("y", s);
    auto xt     = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 2, 3, 1}}}), x);
    auto yt     = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 2, 3, 1}}}), y);
    auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", -1}}), xt, yt);
    auto t = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 2, 3, 1}}}), concat);
    mm->add_return({t});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(p.begin(), p.end()) == n - 2);
    auto new_concat =
        std::find_if(p.begin(), p.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != p.end()});
    EXPECT(migraphx::any_cast<migraphx::op::concat>(new_concat->get_operator()).axis == 1);
}

TEST_CASE(concat_transpose3)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}});
    auto y   = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {1, 5, 3, 4}});
    auto xt  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 2, 3, 1}}}), x);
    auto yt  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 2, 3, 1}}}), y);
    auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 3}}), xt, yt);
    auto t = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 2, 3, 1}}}), concat);
    mm->add_return({t});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(p.begin(), p.end()) == n - 2);
    auto new_concat =
        std::find_if(p.begin(), p.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != p.end()});
    EXPECT(migraphx::any_cast<migraphx::op::concat>(new_concat->get_operator()).axis == 1);
}

TEST_CASE(concat_transpose4)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto sx     = migraphx::shape{migraphx::shape::float_type, {1, 1, 12, 64}};
    auto sy     = migraphx::shape{migraphx::shape::float_type, {1, 12, 1, 64}};
    auto x      = mm->add_parameter("x", sx);
    auto y      = mm->add_parameter("y", sy);
    auto xt     = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 2, 3, 1}}}), x);
    auto yt     = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 3, 2}}}), y);
    auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 3}}), xt, yt);
    auto t = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 2, 3, 1}}}), concat);
    mm->add_return({t});

    migraphx::program p1 = p;
    run_pass(p);

    EXPECT(p1 == p);
}

TEST_CASE(nested_concat)
{
    migraphx::program p;

    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x       = mm->add_parameter("x", s);
    auto y       = mm->add_parameter("y", s);
    auto concat1 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
    auto concat2 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), y, x);
    auto concat3 =
        mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), concat1, concat2);
    mm->add_return({concat3});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(p.begin(), p.end()) == n - 2);
    EXPECT(std::count_if(p.begin(), p.end(), [](auto ins) { return ins.name() == "concat"; }) == 1);
}

TEST_CASE(nested_concat_partial)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x   = mm->add_parameter("x", s);
    auto y   = mm->add_parameter("y", s);
    auto l   = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1, 4, 3, 4}}));
    auto concat1 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
    auto concat2 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), y, x);
    auto concat3 =
        mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), concat1, concat2, l);
    mm->add_return({concat3});
    auto out_shape = p.get_output_shapes().back();
    auto n         = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(p.begin(), p.end()) == n - 2);
    EXPECT(std::count_if(p.begin(), p.end(), [](auto ins) { return ins.name() == "concat"; }) == 1);
}

TEST_CASE(multibroadcast_simplify)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    std::vector<size_t> s_lens{1, 2, 3, 4};
    auto s = migraphx::shape{migraphx::shape::float_type, s_lens};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", s_lens}}), x);
    mm->add_instruction(migraphx::make_op("mul"), y, y);
    auto n = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == n - 1);
}

TEST_CASE(double_slice1)
{
    migraphx::program p1;
    auto* mm1 = p1.get_main_module();
    {
        auto x      = mm1->add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {256}}}), x);
        auto slice2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {64}}}), slice1);
        mm1->add_return({slice2});
    }
    run_pass(p1);

    migraphx::program p2;
    auto* mm2 = p2.get_main_module();
    {
        auto x     = mm2->add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {64}}, {"ends", {96}}}), x);
        mm2->add_return({slice});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(double_slice2)
{
    migraphx::program p1;
    auto* mm1 = p1.get_main_module();
    {
        auto x      = mm1->add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {128}}}), x);
        auto slice2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {32}}}), slice1);
        mm1->add_return({slice2});
    }
    run_pass(p1);

    migraphx::program p2;
    auto* mm2 = p2.get_main_module();
    {
        auto x     = mm2->add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {64}}}), x);
        mm2->add_return({slice});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(double_slice_multi_axes)
{
    migraphx::program p1;
    auto* mm1 = p1.get_main_module();
    {
        auto x      = mm1->add_parameter("x", {migraphx::shape::int32_type, {256, 128}});
        auto slice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {128}}}), x);
        auto slice2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {32}}}), slice1);
        mm1->add_return({slice2});
    }
    run_pass(p1);

    migraphx::program p2;

    auto* mm2 = p2.get_main_module();
    {
        auto x     = mm2->add_parameter("x", {migraphx::shape::int32_type, {256, 128}});
        auto slice = mm2->add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {0, 1}}, {"starts", {32, 0}}, {"ends", {128, 32}}}),
            x);
        mm2->add_return({slice});
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
