#include <migraphx/serialize.hpp>
#include <migraphx/context.hpp>
#include <migraphx/ref/context.hpp>
#include <migraphx/functional.hpp>
#include <test.hpp>

TEST_CASE(context)
{
    migraphx::context ctx = migraphx::ref::context{};
    migraphx::value v     = ctx.to_value();
    EXPECT(v.empty());

    migraphx::context cpu_ctx = migraphx::ref::context{};
    cpu_ctx.from_value(v);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
