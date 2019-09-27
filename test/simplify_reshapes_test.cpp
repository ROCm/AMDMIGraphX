#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct simplify_reshapes_target
{
    std::string name() const { return "simplify_reshapes"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const
    {
        return {migraphx::simplify_reshapes{}, migraphx::dead_code_elimination{}};
    }
    migraphx::context get_context() const { return {}; }
};

TEST_CASE(double_contig)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c1 = p.add_instruction(migraphx::op::contiguous{}, t1);
    auto c2 = p.add_instruction(migraphx::op::contiguous{}, c1);
    p.add_instruction(pass_op{}, c2);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 4);
    auto result = p.eval({});
    EXPECT(result != get_2x2());
}

TEST_CASE(double_transpose)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto t2 = p.add_instruction(migraphx::op::transpose{{1, 0}}, t1);
    p.add_instruction(pass_op{}, t2);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({});
    EXPECT(result == get_2x2());
}

TEST_CASE(double_transpose_contig)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c1 = p.add_instruction(migraphx::op::contiguous{}, t1);
    auto t2 = p.add_instruction(migraphx::op::transpose{{1, 0}}, c1);
    auto c2 = p.add_instruction(migraphx::op::contiguous{}, t2);
    p.add_instruction(pass_op{}, c2);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({});
    EXPECT(result == get_2x2());
}

TEST_CASE(single_transpose)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    p.add_instruction(pass_op{}, t1);
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 3);
    auto result = p.eval({});
    EXPECT(result != get_2x2());
}

TEST_CASE(double_transpose_sin_pass)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    p.add_instruction(migraphx::op::transpose{{1, 0}}, t1);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    // TODO: Fix this
    // EXPECT(std::distance(p.begin(), p.end()) == 1);
    auto result = p.eval({});
    EXPECT(result == get_2x2());
}

TEST_CASE(single_transpose_sin_pass)
{
    migraphx::program p;
    auto l = p.add_literal(get_2x2());
    p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({});
    EXPECT(result != get_2x2());
}

TEST_CASE(reshape_transpose)
{
    migraphx::program p;
    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 112, 56, 56}};
    auto x  = p.add_parameter("x", s);
    auto r1 = p.add_instruction(migraphx::op::reshape{{1, 4, 28, 56, 56}}, x);
    auto t  = p.add_instruction(migraphx::op::transpose{{0, 2, 1, 3, 4}}, r1);
    auto ct = p.add_instruction(migraphx::op::contiguous{}, t);
    auto r2 = p.add_instruction(migraphx::op::reshape{{1, 112, 56, 56}}, ct);
    p.add_instruction(pass_op{}, r2);
    EXPECT(p.get_shape() == s);
    auto n = std::distance(p.begin(), p.end());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape() == s);
    EXPECT(std::distance(p.begin(), p.end()) == n);
}

TEST_CASE(transpose_contiguous)
{
    migraphx::program p;
    auto s  = migraphx::shape{migraphx::shape::float_type, {4, 4}};
    auto x  = p.add_parameter("x", s);
    auto t  = p.add_instruction(migraphx::op::transpose{{1, 0}}, x);
    auto c1 = p.add_instruction(migraphx::op::contiguous{}, t);
    p.add_instruction(pass_op{}, c1);
    auto out_shape = p.get_shape();
    auto n         = std::distance(p.begin(), p.end());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n);
}

TEST_CASE(transpose_double_contiguous)
{
    migraphx::program p;
    auto s  = migraphx::shape{migraphx::shape::float_type, {4, 4}};
    auto x  = p.add_parameter("x", s);
    auto t  = p.add_instruction(migraphx::op::transpose{{1, 0}}, x);
    auto c1 = p.add_instruction(migraphx::op::contiguous{}, t);
    auto c2 = p.add_instruction(migraphx::op::contiguous{}, c1);
    p.add_instruction(pass_op{}, c2);
    auto out_shape = p.get_shape();
    auto n         = std::distance(p.begin(), p.end());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 1);
    EXPECT(p.has_instruction(t));
}

TEST_CASE(transpose_partial1)
{
    migraphx::program p;
    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = p.add_parameter("x", s);
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0, 2}}, x);
    auto t2 = p.add_instruction(migraphx::op::transpose{{1, 2, 0}}, t1);
    p.add_instruction(pass_op{}, t2);
    auto out_shape = p.get_shape();
    auto n         = std::distance(p.begin(), p.end());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 1);
}

TEST_CASE(transpose_partial2)
{
    migraphx::program p;
    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = p.add_parameter("x", s);
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0, 2}}, x);
    auto t2 = p.add_instruction(migraphx::op::transpose{{1, 2, 0}}, t1);
    auto t3 = p.add_instruction(migraphx::op::transpose{{1, 0, 2}}, t2);
    p.add_instruction(pass_op{}, t3);
    auto out_shape = p.get_shape();
    auto n         = std::distance(p.begin(), p.end());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 2);
}

TEST_CASE(transpose_partial3)
{
    migraphx::program p;
    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = p.add_parameter("x", s);
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0, 2}}, x);
    auto t2 = p.add_instruction(migraphx::op::transpose{{1, 2, 0}}, t1);
    auto t3 = p.add_instruction(migraphx::op::transpose{{1, 0, 2}}, t2);
    auto t4 = p.add_instruction(migraphx::op::transpose{{1, 0, 2}}, t3);
    p.add_instruction(pass_op{}, t4);
    auto out_shape = p.get_shape();
    auto n         = std::distance(p.begin(), p.end());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 3);
}

TEST_CASE(nop_transpose1)
{
    migraphx::program p;
    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x = p.add_parameter("x", s);
    auto t = p.add_instruction(migraphx::op::transpose{{0, 1, 2}}, x);
    p.add_instruction(pass_op{}, t);
    auto out_shape = p.get_shape();
    auto n         = std::distance(p.begin(), p.end());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 1);
}

TEST_CASE(nop_transpose2)
{
    migraphx::program p;
    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = p.add_parameter("x", s);
    auto t1 = p.add_instruction(migraphx::op::transpose{{0, 1, 2}}, x);
    auto t2 = p.add_instruction(migraphx::op::transpose{{0, 1, 2}}, t1);
    auto t3 = p.add_instruction(migraphx::op::transpose{{0, 1, 2}}, t2);
    auto t4 = p.add_instruction(migraphx::op::transpose{{0, 1, 2}}, t3);
    p.add_instruction(pass_op{}, t4);
    auto out_shape = p.get_shape();
    auto n         = std::distance(p.begin(), p.end());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 4);
}

TEST_CASE(nop_transpose3)
{
    migraphx::program p;
    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x      = p.add_parameter("x", s);
    auto y      = p.add_parameter("y", s);
    auto concat = p.add_instruction(migraphx::op::concat{3}, x, y);
    auto t1     = p.add_instruction(migraphx::op::transpose{{0, 1, 2, 3}}, concat);
    auto t2     = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, t1);
    p.add_instruction(pass_op{}, t2);
    auto out_shape = p.get_shape();
    auto n         = std::distance(p.begin(), p.end());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape() == out_shape);
    EXPECT(std::distance(p.begin(), p.end()) == n - 1);
}

TEST_CASE(concat_transpose1)
{
    migraphx::program p;
    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x      = p.add_parameter("x", s);
    auto y      = p.add_parameter("y", s);
    auto xt     = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, x);
    auto yt     = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, y);
    auto concat = p.add_instruction(migraphx::op::concat{2}, xt, yt);
    auto t      = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, concat);
    p.add_instruction(pass_op{}, t);
    auto out_shape = p.get_shape();
    auto n         = std::distance(p.begin(), p.end());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape().lens() == out_shape.lens());
    EXPECT(std::distance(p.begin(), p.end()) == n - 3);
    auto new_concat =
        std::find_if(p.begin(), p.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != p.end()});
    EXPECT(migraphx::any_cast<migraphx::op::concat>(new_concat->get_operator()).axis == 3);
}

TEST_CASE(concat_transpose2)
{
    migraphx::program p;
    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x      = p.add_parameter("x", s);
    auto y      = p.add_parameter("y", s);
    auto xt     = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, x);
    auto yt     = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, y);
    auto concat = p.add_instruction(migraphx::op::concat{3}, xt, yt);
    auto t      = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, concat);
    p.add_instruction(pass_op{}, t);
    auto out_shape = p.get_shape();
    auto n         = std::distance(p.begin(), p.end());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape().lens() == out_shape.lens());
    EXPECT(std::distance(p.begin(), p.end()) == n - 2);
    auto new_concat =
        std::find_if(p.begin(), p.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != p.end()});
    EXPECT(migraphx::any_cast<migraphx::op::concat>(new_concat->get_operator()).axis == 1);
}

TEST_CASE(concat_transpose3)
{
    migraphx::program p;
    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x      = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}});
    auto y      = p.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {1, 5, 3, 4}});
    auto xt     = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, x);
    auto yt     = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, y);
    auto concat = p.add_instruction(migraphx::op::concat{3}, xt, yt);
    auto t      = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, concat);
    p.add_instruction(pass_op{}, t);
    auto out_shape = p.get_shape();
    auto n         = std::distance(p.begin(), p.end());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape().lens() == out_shape.lens());
    EXPECT(std::distance(p.begin(), p.end()) == n - 2);
    auto new_concat =
        std::find_if(p.begin(), p.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != p.end()});
    EXPECT(migraphx::any_cast<migraphx::op::concat>(new_concat->get_operator()).axis == 1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
