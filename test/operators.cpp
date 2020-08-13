#include <migraphx/register_op.hpp>
#include <migraphx/operation.hpp>
#include <sstream>
#include <string>
#include "test.hpp"

TEST_CASE(load_op)
{
    for(const auto& name : migraphx::get_operators())
    {
        auto op = migraphx::load_op(name);
        CHECK(op.name() == name);
    }
}

TEST_CASE(ops)
{
    auto names = migraphx::get_operators();
    EXPECT(names.size() > 1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
