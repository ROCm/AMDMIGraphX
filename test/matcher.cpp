#include <migraph/matcher.hpp>
#include <migraph/iterator_for.hpp>
#include <test.hpp>
#include <basic_ops.hpp>

namespace matchers = migraph::matchers;

template<class M>
migraph::matcher_result find_match(migraph::program& p, M&& m)
{
    migraph::matcher_result result;
    for(auto ins:migraph::iterator_for(p))
    {
        result = migraph::match_instruction(p, ins, m);
        if(result.result != p.end())
            return result;
    }
    return result;
}

void match1()
{
    migraph::program p;
    auto l = p.add_literal(1);
    auto m = matchers::standard_shape();
    auto r = find_match(p, m);
    EXPECT(bool{r.result == l});
}

void match_name1()
{
    migraph::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = matchers::name("sum");
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

void match_name2()
{
    migraph::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = matchers::name("min");
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

void match_name3()
{
    migraph::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = matchers::name("sum")(matchers::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

void match_arg1()
{
    migraph::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = matchers::name("sum")(matchers::arg(0)(matchers::name("@literal")), matchers::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

void match_arg2()
{
    migraph::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = matchers::name("sum")(matchers::arg(0)(matchers::name("sum")), matchers::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

void match_arg3()
{
    migraph::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = matchers::name("sum")(matchers::arg(1)(matchers::name("@literal")), matchers::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

void match_arg4()
{
    migraph::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    auto pass = p.add_instruction(pass_op{}, sum);
    auto m = matchers::name("pass")(matchers::arg(0)(matchers::name("sum")), matchers::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == pass});
}

void match_arg5()
{
    migraph::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = matchers::name("pass")(matchers::arg(1)(matchers::name("sum")), matchers::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == p.end()});
}

void match_arg6()
{
    migraph::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = matchers::name("sum")(matchers::arg(0)(matchers::name("@literal")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

void match_arg7()
{
    migraph::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = matchers::name("sum")(matchers::arg(0)(matchers::name("@literal")), matchers::arg(1)(matchers::name("@literal")));
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

void match_args1()
{
    migraph::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(pass_op{}, sum);
    auto m = matchers::name("sum")(matchers::args(matchers::name("@literal"), matchers::name("@literal")), matchers::standard_shape());
    auto r = find_match(p, m);
    EXPECT(bool{r.result == sum});
}

int main() {
    match1();
    match_name1();
    match_name2();
    match_name3();

    match_arg1();
    match_arg2();
    match_arg3();
    match_arg4();
    match_arg5();
    match_arg6();
    match_arg7();

    match_args1();

}
