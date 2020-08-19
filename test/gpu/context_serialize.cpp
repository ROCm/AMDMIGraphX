#include <iostream>
#include <vector>
#include <migraphx/verify.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/context.hpp>
#include "test.hpp"

TEST_CASE(gpu_context)
{
    migraphx::context ctx = migraphx::gpu::context{0, 3};

    auto v = ctx.to_value();
    EXPECT(v.size() == 2);

    EXPECT(v.contains("events"));
    EXPECT(v.at("events").get_uint64() == 0);

    EXPECT(v.contains("streams"));
    EXPECT(v.at("streams").get_uint64() == 3);

    migraphx::gpu::context g_ctx;
    g_ctx.from_value(v);

    auto v1 = g_ctx.to_value();
    EXPECT(v == v1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
