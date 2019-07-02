#include <migraphx/matcher.hpp>
#include <migraphx/iterator_for.hpp>
#include <test.hpp>
#include <basic_ops.hpp>

namespace match = migraphx::match;

template <class M>
migraphx::match::matcher_result find_match(migraphx::program& p, M&& m)
{
    migraphx::match::matcher_result result;
    for(auto ins : migraphx::iterator_for(p))
    {
        result = migraphx::match::match_instruction(p, ins, m);
        if(result.result != p.end())
            return result;
    }
    return result;
}

void match1()
{
    migraphx::program p;
    auto l = p.add_literal(1);
    auto m = match::standard_shape();
    auto r = find_match(p, m);
    EXPECT(bool{r.result == l});
}

TEST_CASE(match_name1)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum");
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_name2)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("min");
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

TEST_CASE(match_name3)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_arg1)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::arg(0)(match::name("@literal")), match::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_arg2)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::arg(0)(match::name("sum")), match::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

TEST_CASE(match_arg3)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::arg(1)(match::name("@literal")), match::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_arg4)
{
    migraphx::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum  = p.add_instruction(sum_op{}, one, two);
    auto pass = p.add_instruction(pass_op{}, sum);
    auto m    = match::name("pass")(match::arg(0)(match::name("sum")), match::standard_shape());
    auto r    = find_match(p, m);
    EXPECT(bool{r.result == pass});
}

TEST_CASE(match_arg5)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("pass")(match::arg(1)(match::name("sum")), match::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

TEST_CASE(match_arg6)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::arg(0)(match::name("@literal")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_arg7)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::arg(0)(match::name("@literal")),
                                match::arg(1)(match::name("@literal")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_arg8)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::all_of(match::arg(0)(match::name("@literal")),
                                match::arg(1)(match::name("@literal"))),
                                match::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_args1)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::args(match::name("@literal"), match::name("@literal")),
                                match::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_args2)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::args(match::name("@literal"), match::name("sum")),
                                match::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

TEST_CASE(match_args3)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::args(match::name("@literal")), match::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

TEST_CASE(match_args4)
{
    migraphx::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    auto sum2 = p.add_instruction(sum_op{}, sum1, two);
    p.add_instruction(pass_op{}, sum2);
    auto m = match::name("sum")(match::args(match::name("sum"), match::name("@literal")),
                                match::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum2});
}

TEST_CASE(match_args5)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::args(match::name("sum"), match::name("@literal")),
                                match::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

TEST_CASE(match_args6)
{
    migraphx::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum  = p.add_instruction(sum_op{}, one, two);
    auto pass = p.add_instruction(pass_op{}, sum);
    auto m    = match::name("pass")(match::args(match::name("sum")), match::standard_shape());
    auto r    = find_match(p, m);
    EXPECT(bool{r.result == pass});
}

TEST_CASE(match_args7)
{
    migraphx::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum  = p.add_instruction(sum_op{}, one, two);
    auto pass = p.add_instruction(pass_op{}, sum);
    auto m    = match::name("pass")(match::args(match::name("sum")(match::args(
                                     match::name("@literal"), match::name("@literal")))),
                                 match::standard_shape());
    auto r    = find_match(p, m);
    EXPECT(bool{r.result == pass});
}

TEST_CASE(match_either_args1)
{
    migraphx::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    auto sum2 = p.add_instruction(sum_op{}, sum1, two);
    p.add_instruction(pass_op{}, sum2);
    auto m =
        match::name("sum")(match::either_arg(0, 1)(match::name("sum"), match::name("@literal")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum2});
}

TEST_CASE(match_either_args2)
{
    migraphx::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    auto sum2 = p.add_instruction(sum_op{}, sum1, two);
    p.add_instruction(pass_op{}, sum2);
    auto m =
        match::name("sum")(match::either_arg(0, 1)(match::name("@literal"), match::name("sum")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum2});
}

TEST_CASE(match_either_args3)
{
    migraphx::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    auto sum2 = p.add_instruction(sum_op{}, sum1, two);
    p.add_instruction(pass_op{}, sum2);
    auto m =
        match::name("sum")(match::either_arg(0, 1)(match::name("pass"), match::name("@literal")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

TEST_CASE(match_all_of1)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::all_of(match::arg(0)(match::name("@literal")),
                                              match::arg(1)(match::name("@literal"))));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_all_of2)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(
        match::all_of(match::arg(0)(match::name("sum")), match::arg(1)(match::name("@literal"))));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

TEST_CASE(match_any_of1)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(
        match::any_of(match::arg(0)(match::name("sum")), match::arg(1)(match::name("@literal"))));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_any_of2)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(
        match::any_of(match::arg(0)(match::name("sum")), match::arg(1)(match::name("sum"))));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

TEST_CASE(match_none_of1)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(
        match::none_of(match::arg(0)(match::name("sum")), match::arg(1)(match::name("sum"))));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

TEST_CASE(match_none_of2)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("sum")(match::none_of(match::arg(0)(match::name("@literal")),
                                               match::arg(1)(match::name("@literal"))));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

TEST_CASE(match_output1)
{
    migraphx::program p;
    auto one   = p.add_literal(1);
    auto two   = p.add_literal(2);
    auto minus = p.add_instruction(minus_op{}, two, one);
    auto sum   = p.add_instruction(sum_op{}, minus, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("minus")(match::output(match::name("sum")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == minus});
}

TEST_CASE(match_output2)
{
    migraphx::program p;
    auto one   = p.add_literal(1);
    auto two   = p.add_literal(2);
    auto minus = p.add_instruction(minus_op{}, two, one);
    auto sum   = p.add_instruction(sum_op{}, minus, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("@literal")(match::output(match::name("sum")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

TEST_CASE(match_skip_output1)
{
    migraphx::program p;
    auto one   = p.add_literal(1);
    auto two   = p.add_literal(2);
    auto minus = p.add_instruction(minus_op{}, two, one);
    auto sum   = p.add_instruction(sum_op{}, minus, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("minus")(match::skip_output(match::name("pass"))(match::name("sum")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == minus});
}

TEST_CASE(match_skip_output2)
{
    migraphx::program p;
    auto one        = p.add_literal(1);
    auto two        = p.add_literal(2);
    auto minus      = p.add_instruction(minus_op{}, two, one);
    auto minus_pass = p.add_instruction(pass_op{}, minus);
    auto sum        = p.add_instruction(sum_op{}, minus_pass, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("minus")(match::skip_output(match::name("pass"))(match::name("sum")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == minus});
}

TEST_CASE(match_skip_output3)
{
    migraphx::program p;
    auto one         = p.add_literal(1);
    auto two         = p.add_literal(2);
    auto minus       = p.add_instruction(minus_op{}, two, one);
    auto minus_pass1 = p.add_instruction(pass_op{}, minus);
    auto minus_pass2 = p.add_instruction(pass_op{}, minus_pass1);
    auto minus_pass3 = p.add_instruction(pass_op{}, minus_pass2);
    auto sum         = p.add_instruction(sum_op{}, minus_pass3, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("minus")(match::skip_output(match::name("pass"))(match::name("sum")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == minus});
}

TEST_CASE(match_skip_output4)
{
    migraphx::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto pass = p.add_instruction(pass_op{}, one);
    auto sum  = p.add_instruction(sum_op{}, pass, two);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("@literal")(match::skip_output(match::name("pass"))(match::name("sum")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == two});
}

TEST_CASE(match_skip_output5)
{
    migraphx::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto pass = p.add_instruction(pass_op{}, one);
    auto sum1 = p.add_instruction(sum_op{}, pass, two);
    auto sum2 = p.add_instruction(sum_op{}, sum1, one);
    auto sum3 = p.add_instruction(sum_op{}, sum2, two);
    p.add_instruction(pass_op{}, sum3);
    auto m = match::name("@literal")(match::skip_output(match::name("pass"))(match::name("sum")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

TEST_CASE(match_skip_output6)
{
    migraphx::program p;
    auto one   = p.add_literal(1);
    auto two   = p.add_literal(2);
    auto minus = p.add_instruction(minus_op{}, two, one);
    auto sum1  = p.add_instruction(sum_op{}, minus, two);
    auto sum2  = p.add_instruction(sum_op{}, sum1, one);
    auto sum3  = p.add_instruction(sum_op{}, sum2, two);
    p.add_instruction(pass_op{}, sum3);
    auto m = match::name("minus")(match::skip_output(match::name("pass"))(match::name("sum")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == minus});
}

TEST_CASE(match_skip_output7)
{
    migraphx::program p;
    auto one    = p.add_literal(1);
    auto two    = p.add_literal(2);
    auto minus1 = p.add_instruction(minus_op{}, two, one);
    auto minus2 = p.add_instruction(minus_op{}, two, minus1);
    auto sum    = p.add_instruction(sum_op{}, one, minus2);
    p.add_instruction(pass_op{}, sum);
    auto m = match::name("minus")(match::skip_output(match::name("pass"))(match::name("minus")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == minus1});
}

TEST_CASE(match_bind1)
{
    migraphx::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum  = p.add_instruction(sum_op{}, one, two);
    auto pass = p.add_instruction(pass_op{}, sum);
    auto m    = match::name("pass")(
                 match::args(match::name("sum")(match::args(match::name("@literal").bind("one"),
                                                            match::name("@literal").bind("two")))
                                 .bind("sum")),
                 match::standard_shape())
                 .bind("pass");
    auto r = find_match(p, m);
    EXPECT(bool{r.instructions.at("one") == one});
    EXPECT(bool{r.instructions.at("two") == two});
    EXPECT(bool{r.instructions.at("sum") == sum});
    EXPECT(bool{r.instructions.at("pass") == pass});
    EXPECT(bool{r.result == pass});
}

struct match_find_sum
{
    migraphx::instruction_ref ins;
    auto matcher() const { return match::name("sum"); }

    void apply(migraphx::program&, const match::matcher_result& r) const
    {
        EXPECT(bool{r.result == ins});
    }
};

struct match_find_literal
{
    migraphx::instruction_ref ins;
    auto matcher() const { return match::name("@literal"); }

    void apply(migraphx::program&, const match::matcher_result& r) const
    {
        EXPECT(bool{r.result != ins});
        EXPECT(r.result->name() == "@literal");
    }
};

TEST_CASE(match_finder)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    match::find_matches(p, match_find_sum{sum}, match_find_literal{sum});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
