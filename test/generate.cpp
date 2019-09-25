#include <migraphx/generate.hpp>
#include "test.hpp"

TEST_CASE(generate)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 4, 1, 1}};
    EXPECT(migraphx::generate_literal(s, 1) == migraphx::generate_argument(s, 1));
    EXPECT(migraphx::generate_literal(s, 1) != migraphx::generate_argument(s, 0));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
