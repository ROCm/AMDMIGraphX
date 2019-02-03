#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/tf.hpp>
#include "test.hpp"

TEST_CASE(relu_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_instruction(migraphx::op::relu{}, l0);
    auto prog = migraphx::parse_tf("relu_test.pb", false);

    EXPECT(p == prog);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
