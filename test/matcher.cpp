#include <migraphx/matcher.hpp>
#include <migraphx/iterator_for.hpp>
#include <test.hpp>
#include <basic_ops.hpp>

namespace match = migraphx::match;

MIGRAPHX_PRED_MATCHER(throws, migraphx::instruction_ref) { MIGRAPHX_THROW("Matcher throws"); }

template <class M>
migraphx::match::matcher_result find_match(migraphx::module& modl, M&& m)
{
    migraphx::match::matcher_result result;
    for(auto ins : migraphx::iterator_for(modl))
    {
        result = migraphx::match::match_instruction(modl, ins, m);
        if(result.result != modl.end())
            return result;
    }
    return result;
}

void match1()
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(1);
    auto m   = match::standard_shape();
    auto r   = find_match(*mm, m);
    EXPECT(bool{r.result == l});
}

TEST_CASE(match_name1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum");
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_name2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("min");
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_name3)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::standard_shape());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_arg1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::arg(0)(match::name("@literal")), match::standard_shape());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_arg2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::arg(0)(match::name("sum")), match::standard_shape());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_arg3)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::arg(1)(match::name("@literal")), match::standard_shape());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_arg4)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum  = mm->add_instruction(sum_op{}, one, two);
    auto pass = mm->add_instruction(pass_op{}, sum);
    auto m    = match::name("pass")(match::arg(0)(match::name("sum")), match::standard_shape());
    auto r    = find_match(*mm, m);
    EXPECT(bool{r.result == pass});
}

TEST_CASE(match_arg5)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("pass")(match::arg(1)(match::name("sum")), match::standard_shape());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_arg6)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::arg(0)(match::name("@literal")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_arg7)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::arg(0)(match::name("@literal")),
                                match::arg(1)(match::name("@literal")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_arg8)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::all_of(match::arg(0)(match::name("@literal")),
                                              match::arg(1)(match::name("@literal"))),
                                match::standard_shape());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_nargs1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::nargs(2));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_nargs2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::nargs(2), match::standard_shape());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_nargs3)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::all_of(match::nargs(2)));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_args1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::args(match::name("@literal"), match::name("@literal")),
                                match::standard_shape());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_args2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::args(match::name("@literal"), match::name("sum")),
                                match::standard_shape());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_args3)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::args(match::name("@literal")), match::standard_shape());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_args4)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::name("sum")(match::args(match::name("sum"), match::name("@literal")),
                                match::standard_shape());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum2});
}

TEST_CASE(match_args5)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::args(match::name("sum"), match::name("@literal")),
                                match::standard_shape());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_args6)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum  = mm->add_instruction(sum_op{}, one, two);
    auto pass = mm->add_instruction(pass_op{}, sum);
    auto m    = match::name("pass")(match::args(match::name("sum")), match::standard_shape());
    auto r    = find_match(*mm, m);
    EXPECT(bool{r.result == pass});
}

TEST_CASE(match_args7)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum  = mm->add_instruction(sum_op{}, one, two);
    auto pass = mm->add_instruction(pass_op{}, sum);
    auto m    = match::name("pass")(match::args(match::name("sum")(match::args(
                                     match::name("@literal"), match::name("@literal")))),
                                 match::standard_shape());
    auto r    = find_match(*mm, m);
    EXPECT(bool{r.result == pass});
}

TEST_CASE(match_either_args1)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m =
        match::name("sum")(match::either_arg(0, 1)(match::name("sum"), match::name("@literal")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum2});
}

TEST_CASE(match_either_args2)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m =
        match::name("sum")(match::either_arg(0, 1)(match::name("@literal"), match::name("sum")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum2});
}

TEST_CASE(match_either_args3)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m =
        match::name("sum")(match::either_arg(0, 1)(match::name("pass"), match::name("@literal")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_either_args_any1)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m =
        match::name("sum")(match::either_arg(0, 1)(match::any().bind("x"), match::any().bind("y")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum1});
    EXPECT(bool{r.instructions.at("x") != r.instructions.at("y")});
}

TEST_CASE(match_either_args_any2)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::name("sum")(
        match::either_arg(0, 1)(match::any().bind("x"), match::name("@literal").bind("y")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum1});
    EXPECT(bool{r.instructions.at("x") != r.instructions.at("y")});
}

TEST_CASE(match_either_args_any3)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::name("sum")(
        match::either_arg(0, 1)(match::name("@literal").bind("x"), match::any().bind("y")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum1});
    EXPECT(bool{r.instructions.at("x") != r.instructions.at("y")});
}

TEST_CASE(match_either_args_any4)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::name("sum")(
        match::either_arg(0, 1)(match::name("sum").bind("x"), match::any().bind("y")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum2});
    EXPECT(bool{r.instructions.at("x") != r.instructions.at("y")});
}

TEST_CASE(match_either_args_any5)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::name("sum")(
        match::either_arg(0, 1)(match::any().bind("x"), match::name("sum").bind("y")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum2});
    EXPECT(bool{r.instructions.at("x") != r.instructions.at("y")});
}

TEST_CASE(match_all_of1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::all_of(match::arg(0)(match::name("@literal")),
                                              match::arg(1)(match::name("@literal"))));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_all_of2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(
        match::all_of(match::arg(0)(match::name("sum")), match::arg(1)(match::name("@literal"))));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_all_of3)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::all_of(match::all_of(
        match::arg(0)(match::name("@literal")), match::arg(1)(match::name("@literal")))));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_lazy_any_of)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    mm->add_instruction(pass_op{}, one);
    auto m = match::any_of(match::any(), throws());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == one});
}

TEST_CASE(match_lazy_all_of)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    mm->add_instruction(pass_op{}, one);
    auto m = match::all_of(match::none(), throws());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_lazy_none_of)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    mm->add_instruction(pass_op{}, one);
    auto m = match::none_of(match::any(), throws());
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_any_of1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(
        match::any_of(match::arg(0)(match::name("sum")), match::arg(1)(match::name("@literal"))));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_any_of2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(
        match::any_of(match::arg(0)(match::name("sum")), match::arg(1)(match::name("sum"))));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_any_of_lazy1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(
        match::any_of(match::args(match::any(), match::any()).bind("x"),
                      match::args(match::name("sum"), match::name("sum")).bind("y")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
    EXPECT(migraphx::contains(r.instructions, "x"));
    EXPECT(bool{r.instructions["x"] == sum});
    EXPECT(not migraphx::contains(r.instructions, "y"));
}

TEST_CASE(match_any_of_lazy2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(
        match::any_of(match::args(match::name("@literal"), match::name("@literal")).bind("x"),
                      match::args(match::any(), match::any()).bind("y")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
    EXPECT(migraphx::contains(r.instructions, "x"));
    EXPECT(bool{r.instructions["x"] == sum});
    EXPECT(not migraphx::contains(r.instructions, "y"));
}

TEST_CASE(match_any_of_lazy3)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(
        match::any_of(match::args(match::any(), match::any()).bind("x"),
                      match::args(match::name("@literal"), match::name("@literal")).bind("y")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
    EXPECT(migraphx::contains(r.instructions, "x"));
    EXPECT(bool{r.instructions["x"] == sum});
    EXPECT(not migraphx::contains(r.instructions, "y"));
}

TEST_CASE(match_any_of_lazy4)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::any_of(
        match::args(match::name("@literal").bind("x1"), match::name("@literal").bind("y1")),
        match::args(match::any().bind("x2"), match::any().bind("y2"))));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
    EXPECT(migraphx::contains(r.instructions, "x1"));
    EXPECT(migraphx::contains(r.instructions, "y1"));
    EXPECT(bool{r.instructions["x1"] == one});
    EXPECT(bool{r.instructions["y1"] == two});
    EXPECT(not migraphx::contains(r.instructions, "x2"));
    EXPECT(not migraphx::contains(r.instructions, "y2"));
}

TEST_CASE(match_any_of_lazy5)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::any_of(
        match::args(match::any().bind("x1"), match::any().bind("y1")),
        match::args(match::name("@literal").bind("x2"), match::name("@literal").bind("y2"))));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
    EXPECT(migraphx::contains(r.instructions, "x1"));
    EXPECT(migraphx::contains(r.instructions, "y1"));
    EXPECT(bool{r.instructions["x1"] == one});
    EXPECT(bool{r.instructions["y1"] == two});
    EXPECT(not migraphx::contains(r.instructions, "x2"));
    EXPECT(not migraphx::contains(r.instructions, "y2"));
}

TEST_CASE(match_none_of1)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(
        match::none_of(match::arg(0)(match::name("sum")), match::arg(1)(match::name("sum"))));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_none_of2)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::none_of(match::arg(0)(match::name("@literal")),
                                               match::arg(1)(match::name("@literal"))));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_output1)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto minus = mm->add_instruction(minus_op{}, two, one);
    auto sum   = mm->add_instruction(sum_op{}, minus, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("minus")(match::output(match::name("sum")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == minus});
}

TEST_CASE(match_output2)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto minus = mm->add_instruction(minus_op{}, two, one);
    auto sum   = mm->add_instruction(sum_op{}, minus, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("@literal")(match::output(match::name("sum")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_skip_output1)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto minus = mm->add_instruction(minus_op{}, two, one);
    auto sum   = mm->add_instruction(sum_op{}, minus, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("minus")(match::skip_output(match::name("pass"))(match::name("sum")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == minus});
}

TEST_CASE(match_skip_output2)
{
    migraphx::program p;

    auto* mm        = p.get_main_module();
    auto one        = mm->add_literal(1);
    auto two        = mm->add_literal(2);
    auto minus      = mm->add_instruction(minus_op{}, two, one);
    auto minus_pass = mm->add_instruction(pass_op{}, minus);
    auto sum        = mm->add_instruction(sum_op{}, minus_pass, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("minus")(match::skip_output(match::name("pass"))(match::name("sum")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == minus});
}

TEST_CASE(match_skip_output3)
{
    migraphx::program p;

    auto* mm         = p.get_main_module();
    auto one         = mm->add_literal(1);
    auto two         = mm->add_literal(2);
    auto minus       = mm->add_instruction(minus_op{}, two, one);
    auto minus_pass1 = mm->add_instruction(pass_op{}, minus);
    auto minus_pass2 = mm->add_instruction(pass_op{}, minus_pass1);
    auto minus_pass3 = mm->add_instruction(pass_op{}, minus_pass2);
    auto sum         = mm->add_instruction(sum_op{}, minus_pass3, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("minus")(match::skip_output(match::name("pass"))(match::name("sum")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == minus});
}

TEST_CASE(match_skip_output4)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto pass = mm->add_instruction(pass_op{}, one);
    auto sum  = mm->add_instruction(sum_op{}, pass, two);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("@literal")(match::skip_output(match::name("pass"))(match::name("sum")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == two});
}

TEST_CASE(match_skip_output5)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto pass = mm->add_instruction(pass_op{}, one);
    auto sum1 = mm->add_instruction(sum_op{}, pass, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, one);
    auto sum3 = mm->add_instruction(sum_op{}, sum2, two);
    mm->add_instruction(pass_op{}, sum3);
    auto m = match::name("@literal")(match::skip_output(match::name("pass"))(match::name("sum")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_skip_output6)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto minus = mm->add_instruction(minus_op{}, two, one);
    auto sum1  = mm->add_instruction(sum_op{}, minus, two);
    auto sum2  = mm->add_instruction(sum_op{}, sum1, one);
    auto sum3  = mm->add_instruction(sum_op{}, sum2, two);
    mm->add_instruction(pass_op{}, sum3);
    auto m = match::name("minus")(match::skip_output(match::name("pass"))(match::name("sum")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == minus});
}

TEST_CASE(match_skip_output7)
{
    migraphx::program p;

    auto* mm    = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto two    = mm->add_literal(2);
    auto minus1 = mm->add_instruction(minus_op{}, two, one);
    auto minus2 = mm->add_instruction(minus_op{}, two, minus1);
    auto sum    = mm->add_instruction(sum_op{}, one, minus2);
    mm->add_instruction(pass_op{}, sum);
    auto m = match::name("minus")(match::skip_output(match::name("pass"))(match::name("minus")));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == minus1});
}

TEST_CASE(match_bind1)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum  = mm->add_instruction(sum_op{}, one, two);
    auto pass = mm->add_instruction(pass_op{}, sum);
    auto m    = match::name("pass")(
                 match::args(match::name("sum")(match::args(match::name("@literal").bind("one"),
                                                            match::name("@literal").bind("two")))
                                 .bind("sum")),
                 match::standard_shape())
                 .bind("pass");
    auto r = find_match(*mm, m);
    EXPECT(bool{r.instructions.at("one") == one});
    EXPECT(bool{r.instructions.at("two") == two});
    EXPECT(bool{r.instructions.at("sum") == sum});
    EXPECT(bool{r.instructions.at("pass") == pass});
    EXPECT(bool{r.result == pass});
}

TEST_CASE(match_has_value1)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::has_value(1);
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == one});
}

TEST_CASE(match_has_value2)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::has_value(2);
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == two});
}

TEST_CASE(match_has_value3)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::name("sum")(match::args(match::has_value(1), match::has_value(2)));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum1});
}

TEST_CASE(match_has_value4)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::has_value(3);
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_has_value5)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::name("sum")(match::args(match::has_value(1), match::has_value(3)));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_has_value6)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(sum_op{}, one, two);
    auto sum2 = mm->add_instruction(sum_op{}, sum1, two);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::name("sum")(match::args(match::has_value(2), match::has_value(1)));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_tree1)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto three = mm->add_literal(3);
    auto sum1  = mm->add_instruction(sum_op{}, one, two);
    auto sum2  = mm->add_instruction(sum_op{}, sum1, three);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::tree("sum", match::has_value(1), match::has_value(2), match::has_value(3));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum2});
}

TEST_CASE(match_tree2)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto three = mm->add_literal(3);
    auto sum1  = mm->add_instruction(sum_op{}, one, two);
    auto sum2  = mm->add_instruction(sum_op{}, sum1, three);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::tree("sum", match::has_value(2), match::has_value(1), match::has_value(3));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_tree3)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto three = mm->add_literal(3);
    auto sum1  = mm->add_instruction(sum_op{}, one, two);
    auto sum2  = mm->add_instruction(sum_op{}, three, sum1);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::tree("sum", match::has_value(3), match::has_value(1), match::has_value(2));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum2});
}

TEST_CASE(match_tree4)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto three = mm->add_literal(3);
    auto sum1  = mm->add_instruction(sum_op{}, one, two);
    auto sum2  = mm->add_instruction(sum_op{}, sum1, three);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::tree(
        "sum", match::has_value(1), match::has_value(2), match::has_value(3), match::has_value(4));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_tree5)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto three = mm->add_literal(3);
    auto sum1  = mm->add_instruction(sum_op{}, one, two);
    auto sum2  = mm->add_instruction(sum_op{}, sum1, three);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::tree("sum", match::has_value(2), match::has_value(3));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_tree6)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto three = mm->add_literal(3);
    auto sum1  = mm->add_instruction(sum_op{}, one, two);
    auto sum2  = mm->add_instruction(sum_op{}, sum1, three);
    mm->add_instruction(pass_op{}, sum2);
    auto m = match::tree("sum", match::has_value(1), match::has_value(3));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

TEST_CASE(match_unordered_tree1)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto three = mm->add_literal(3);
    auto sum1  = mm->add_instruction(sum_op{}, one, two);
    auto sum2  = mm->add_instruction(sum_op{}, sum1, three);
    mm->add_instruction(pass_op{}, sum2);
    auto m =
        match::unordered_tree("sum", match::has_value(3), match::has_value(2), match::has_value(1));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum2});
}

TEST_CASE(match_unordered_tree2)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto three = mm->add_literal(3);
    auto sum1  = mm->add_instruction(sum_op{}, one, two);
    auto sum2  = mm->add_instruction(sum_op{}, three, sum1);
    mm->add_instruction(pass_op{}, sum2);
    auto m =
        match::unordered_tree("sum", match::has_value(3), match::has_value(2), match::has_value(1));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum2});
}

TEST_CASE(match_unordered_tree3)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto three = mm->add_literal(3);
    auto sum1  = mm->add_instruction(sum_op{}, two, one);
    auto sum2  = mm->add_instruction(sum_op{}, sum1, three);
    mm->add_instruction(pass_op{}, sum2);
    auto m =
        match::unordered_tree("sum", match::has_value(3), match::has_value(2), match::has_value(1));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == sum2});
}

TEST_CASE(match_unordered_tree4)
{
    migraphx::program p;

    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto three = mm->add_literal(3);
    auto sum1  = mm->add_instruction(sum_op{}, one, two);
    auto sum2  = mm->add_instruction(sum_op{}, sum1, three);
    mm->add_instruction(pass_op{}, sum2);
    auto m =
        match::unordered_tree("sum", match::has_value(4), match::has_value(2), match::has_value(1));
    auto r = find_match(*mm, m);
    EXPECT(bool{r.result == mm->end()});
}

struct match_find_sum
{
    migraphx::instruction_ref ins;
    auto matcher() const { return match::name("sum"); }

    void apply(migraphx::module&, const match::matcher_result& r) const
    {
        EXPECT(bool{r.result == ins});
    }
};

struct match_find_literal
{
    migraphx::instruction_ref ins;
    auto matcher() const { return match::name("@literal"); }

    void apply(migraphx::module&, const match::matcher_result& r) const
    {
        EXPECT(bool{r.result != ins});
        EXPECT(r.result->name() == "@literal");
    }
};

TEST_CASE(match_finder)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(pass_op{}, sum);
    match::find_matches(*mm, match_find_sum{sum}, match_find_literal{sum});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
