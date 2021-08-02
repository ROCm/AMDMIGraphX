#include <migraphx/generate.hpp>
#include "test.hpp"

TEST_CASE(generate)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 4, 1, 1}};
    EXPECT(migraphx::generate_literal(s, 1) == migraphx::generate_argument(s, 1));
    EXPECT(migraphx::generate_literal(s, 1) != migraphx::generate_argument(s, 0));
}

TEST_CASE(fill_tuple)
{
    migraphx::shape s0{migraphx::shape::float_type, {4, 4, 1, 1}};
    migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::bool_type, {3, 2}};
    migraphx::shape s({s0, s1, s2});
    auto arg         = migraphx::fill_argument(s, 1);
    const auto& args = arg.get_sub_objects();
    EXPECT(args.at(0) == migraphx::fill_argument(s0, 1));
    EXPECT(args.at(1) == migraphx::fill_argument(s1, 1));
    EXPECT(args.at(2) == migraphx::fill_argument(s2, 1));
}

TEST_CASE(generate_tuple)
{
    migraphx::shape s0{migraphx::shape::float_type, {4, 4, 1, 1}};
    migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::bool_type, {3, 2}};
    migraphx::shape s({s0, s1, s2});
    auto arg         = migraphx::generate_argument(s, 1);
    const auto& args = arg.get_sub_objects();
    EXPECT(args.at(0) == migraphx::generate_argument(s0, 1));
    EXPECT(args.at(1) == migraphx::generate_argument(s1, 1));
    EXPECT(args.at(2) == migraphx::generate_argument(s2, 1));

    EXPECT(args.at(0) != migraphx::generate_argument(s0, 0));
    EXPECT(args.at(1) != migraphx::generate_argument(s1, 2));
    EXPECT(args.at(2) != migraphx::generate_argument(s2, 0));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
