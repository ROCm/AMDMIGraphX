#include <migraphx/serialize.hpp>
#include <migraphx/context.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/functional.hpp>
#include <test.hpp>

TEST_CASE(context)
{
    migraphx::context ctx = migraphx::cpu::context{};
    migraphx::value v     = ctx.to_value();
    EXPECT(v.empty());

    migraphx::cpu::context cpu_ctx;
    cpu_ctx.from_value(v);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
