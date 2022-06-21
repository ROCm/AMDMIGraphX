#include <iostream>
#include <vector>
#include <migraphx/verify.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/context.hpp>
#include "test.hpp"

TEST_CASE(gpu_context_serialize)
{
    migraphx::context ctx = migraphx::gpu::context{0, 3};

    auto v = ctx.to_value();
    EXPECT(v.size() == 2);

    EXPECT(v.contains("events"));
    EXPECT(v.at("events").without_key().to<std::size_t>() == 0);

    EXPECT(v.contains("streams"));
    EXPECT(v.at("streams").without_key().to<std::size_t>() == 3);

    migraphx::gpu::context g_ctx;
    g_ctx.from_value(v);

    auto v1 = g_ctx.to_value();
    EXPECT(v == v1);
}

TEST_CASE(context_queue)
{
    migraphx::context ctx = migraphx::gpu::context{0, 3};
    EXPECT(ctx.get_queue().get<hipStream_t>() != nullptr);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
