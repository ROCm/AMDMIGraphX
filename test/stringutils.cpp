#include <migraphx/stringutils.hpp>
#include <test.hpp>

TEST_CASE(interpolate_string_simple1)
{
    std::string input = "Hello ${w}!";
    auto s            = migraphx::interpolate_string(input, {{"w", "world"}});
    EXPECT(s == "Hello world!");
}

TEST_CASE(interpolate_string_simple2)
{
    std::string input = "${hello}";
    auto s            = migraphx::interpolate_string(input, {{"hello", "bye"}});
    EXPECT(s == "bye");
}

TEST_CASE(interpolate_string_unbalanced)
{
    std::string input = "${hello";
    EXPECT(test::throws([&] { migraphx::interpolate_string(input, {{"hello", "bye"}}); }));
}

TEST_CASE(interpolate_string_extra_space)
{
    std::string input = "${  hello  }";
    auto s            = migraphx::interpolate_string(input, {{"hello", "bye"}});
    EXPECT(s == "bye");
}

TEST_CASE(interpolate_string_multiple)
{
    std::string input = "${h} ${w}!";
    auto s            = migraphx::interpolate_string(input, {{"w", "world"}, {"h", "Hello"}});
    EXPECT(s == "Hello world!");
}

TEST_CASE(interpolate_string_next)
{
    std::string input = "${hh}${ww}!";
    auto s            = migraphx::interpolate_string(input, {{"ww", "world"}, {"hh", "Hello"}});
    EXPECT(s == "Helloworld!");
}

TEST_CASE(interpolate_string_dollar_sign)
{
    std::string input = "$hello";
    auto s            = migraphx::interpolate_string(input, {{"hello", "bye"}});
    EXPECT(s == "$hello");
}

TEST_CASE(interpolate_string_missing)
{
    std::string input = "${hello}";
    EXPECT(test::throws([&] { migraphx::interpolate_string(input, {{"h", "bye"}}); }));
}

TEST_CASE(interpolate_string_custom1)
{
    std::string input = "****{{a}}****";
    auto s            = migraphx::interpolate_string(input, {{"a", "b"}}, "{{", "}}");
    EXPECT(s == "****b****");
}

TEST_CASE(interpolate_string_custom2)
{
    std::string input = "****{{{a}}}****";
    auto s            = migraphx::interpolate_string(input, {{"a", "b"}}, "{{{", "}}}");
    EXPECT(s == "****b****");
}

TEST_CASE(interpolate_string_custom3)
{
    std::string input = "****{{{{a}}}}****";
    auto s            = migraphx::interpolate_string(input, {{"a", "b"}}, "{{{{", "}}}}");
    EXPECT(s == "****b****");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
