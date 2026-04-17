/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <iostream>
#include <vector>
#include <migraphx/register_target.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/gpu/target.hpp>
#include "test.hpp"

using hip_stream_ptr = MIGRAPHX_MANAGE_PTR(hipStream_t, hipStreamDestroy);

static hip_stream_ptr create_external_stream()
{
    hipStream_t stream;
    auto status = hipStreamCreate(&stream);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to create stream");
    return hip_stream_ptr{stream};
}

static void verify_data(const migraphx::argument& result, const migraphx::shape& s, float expected)
{
    std::vector<float> expected_data(s.elements(), expected);
    auto expected_arg = migraphx::argument{s, expected_data.data()};
    EXPECT(result == expected_arg);
}

TEST_CASE(test_stream_override_get)
{
    migraphx::gpu::context ctx{};
    auto& stream = ctx.get_stream();

    hipStream_t internal = stream.get();
    EXPECT(internal != nullptr);

    auto ext = create_external_stream();
    stream.set_external_stream(ext.get());

    EXPECT(stream.get() == ext.get());
    EXPECT(stream.get() != internal);
    EXPECT(stream.has_external_stream());

    stream.set_external_stream(nullptr);

    EXPECT(stream.get() == internal);
    EXPECT(not stream.has_external_stream());
}

TEST_CASE(test_stream_override_get_queue)
{
    migraphx::gpu::context ctx{};
    auto ext = create_external_stream();

    hipStream_t original_queue = ctx.get_queue().get<hipStream_t>();
    EXPECT(original_queue != nullptr);

    ctx.get_stream().set_external_stream(ext.get());
    EXPECT(ctx.get_queue().get<hipStream_t>() == ext.get());

    ctx.get_stream().set_external_stream(nullptr);

    EXPECT(ctx.get_queue().get<hipStream_t>() == original_queue);
}

TEST_CASE(test_context_wait_for_sets_external_stream)
{
    migraphx::gpu::context ctx{};
    auto ext = create_external_stream();

    migraphx::any_ptr queue(ext.get());

    hipStream_t before = ctx.get_queue().get<hipStream_t>();
    ctx.wait_for(queue);
    EXPECT(ctx.get_queue().get<hipStream_t>() == ext.get());
    EXPECT(ctx.get_queue().get<hipStream_t>() != before);

    ctx.finish_on(queue);
    EXPECT(ctx.get_queue().get<hipStream_t>() == before);
}

TEST_CASE(test_external_stream_eval_uses_caller_stream)
{
    const unsigned int m = 64;
    const unsigned int k = 128;

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {m, k}});
    auto y = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {k, m}}));
    mm->add_instruction(migraphx::make_op("dot"), x, y);

    p.compile(migraphx::make_target("gpu"));

    migraphx::shape input_shape{migraphx::shape::float_type, {m, k}};
    migraphx::shape output_shape{migraphx::shape::float_type, {m, m}};
    auto input  = migraphx::fill_argument(input_shape, 1);
    auto ginput = migraphx::gpu::to_gpu(input);

    auto output  = migraphx::fill_argument(output_shape, 0);
    auto goutput = migraphx::gpu::to_gpu(output);

    auto ext = create_external_stream();

    auto results = p.eval({{"x", ginput}, {"main:#output_0", goutput}}, {ext.get(), true});

    EXPECT(not results.empty());

    EXPECT(hipStreamSynchronize(ext.get()) == hipSuccess);
    auto host_output = migraphx::gpu::from_gpu(goutput);
    EXPECT(host_output != output);
}

TEST_CASE(test_external_stream_serialized_on_caller_stream)
{
    const unsigned int n = 256;

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {n}});
    auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {n}});
    mm->add_instruction(migraphx::make_op("add"), x, y);

    p.compile(migraphx::make_target("gpu"));

    std::vector<float> xdata(n, 1.0f);
    std::vector<float> ydata(n, 2.0f);
    auto xarg = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n}}, xdata.data()};
    auto yarg = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n}}, ydata.data()};

    auto gx = migraphx::gpu::to_gpu(xarg);
    auto gy = migraphx::gpu::to_gpu(yarg);

    migraphx::shape out_shape{migraphx::shape::float_type, {n}};
    auto out  = migraphx::fill_argument(out_shape, 0);
    auto gout = migraphx::gpu::to_gpu(out);

    auto ext = create_external_stream();

    auto results = p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout}}, {ext.get(), true});

    EXPECT(not results.empty());

    EXPECT(hipStreamSynchronize(ext.get()) == hipSuccess);
    auto host_result = migraphx::gpu::from_gpu(gout);
    verify_data(host_result, out_shape, 3.0f);
}

TEST_CASE(test_multiple_async_evals_same_stream)
{
    const unsigned int n = 128;

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {n}});
    auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {n}});
    mm->add_instruction(migraphx::make_op("add"), x, y);

    p.compile(migraphx::make_target("gpu"));

    std::vector<float> xdata(n, 1.0f);
    std::vector<float> ydata(n, 2.0f);
    auto xarg = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n}}, xdata.data()};
    auto yarg = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n}}, ydata.data()};

    auto gx = migraphx::gpu::to_gpu(xarg);
    auto gy = migraphx::gpu::to_gpu(yarg);

    migraphx::shape out_shape{migraphx::shape::float_type, {n}};
    auto out  = migraphx::fill_argument(out_shape, 0);
    auto gout = migraphx::gpu::to_gpu(out);

    auto ext = create_external_stream();

    for(int iter = 0; iter < 5; ++iter)
    {
        p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout}}, {ext.get(), true});
    }

    EXPECT(hipStreamSynchronize(ext.get()) == hipSuccess);
    auto host_result = migraphx::gpu::from_gpu(gout);
    verify_data(host_result, out_shape, 3.0f);
}

TEST_CASE(test_external_stream_cleared_after_eval)
{
    const unsigned int n = 64;

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {n}});
    auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {n}});
    mm->add_instruction(migraphx::make_op("add"), x, y);

    p.compile(migraphx::make_target("gpu"));

    std::vector<float> xdata(n, 1.0f);
    std::vector<float> ydata(n, 2.0f);
    auto xarg = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n}}, xdata.data()};
    auto yarg = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n}}, ydata.data()};

    auto gx = migraphx::gpu::to_gpu(xarg);
    auto gy = migraphx::gpu::to_gpu(yarg);

    migraphx::shape out_shape{migraphx::shape::float_type, {n}};
    auto out  = migraphx::fill_argument(out_shape, 0);
    auto gout = migraphx::gpu::to_gpu(out);

    auto ext = create_external_stream();

    migraphx::context& ctx_ref = p.get_context();
    auto* gpu_ctx              = ctx_ref.any_cast<migraphx::gpu::context>();
    EXPECT(gpu_ctx != nullptr);

    hipStream_t internal_stream = gpu_ctx->get_queue().get<hipStream_t>();

    p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout}}, {ext.get(), true});

    EXPECT(gpu_ctx->get_queue().get<hipStream_t>() == internal_stream);
    EXPECT(not gpu_ctx->get_stream().has_external_stream());
}

TEST_CASE(test_wait_for_null_stream_uses_event_fallback)
{
    migraphx::gpu::context ctx{};

    migraphx::any_ptr queue{};

    hipStream_t internal_before = ctx.get_queue().get<hipStream_t>();

    ctx.wait_for(queue);

    EXPECT(not ctx.get_stream().has_external_stream());
    EXPECT(ctx.get_queue().get<hipStream_t>() == internal_before);

    ctx.finish_on(queue);

    EXPECT(not ctx.get_stream().has_external_stream());
    EXPECT(ctx.get_queue().get<hipStream_t>() == internal_before);
}

TEST_CASE(test_fallback_event_path_produces_correct_results)
{
    const unsigned int n = 128;

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {n}});
    auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {n}});
    mm->add_instruction(migraphx::make_op("add"), x, y);

    p.compile(migraphx::make_target("gpu"));

    std::vector<float> xdata(n, 5.0f);
    std::vector<float> ydata(n, 7.0f);
    auto xarg = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n}}, xdata.data()};
    auto yarg = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n}}, ydata.data()};

    auto gx = migraphx::gpu::to_gpu(xarg);
    auto gy = migraphx::gpu::to_gpu(yarg);

    migraphx::shape out_shape{migraphx::shape::float_type, {n}};
    auto out  = migraphx::fill_argument(out_shape, 0);
    auto gout = migraphx::gpu::to_gpu(out);

    auto results =
        p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout}}, {migraphx::any_ptr{}, true});

    EXPECT(not results.empty());

    EXPECT(hipDeviceSynchronize() == hipSuccess);
    auto host_result = migraphx::gpu::from_gpu(gout);
    verify_data(host_result, out_shape, 12.0f);
}

TEST_CASE(test_non_async_eval_uses_internal_stream)
{
    const unsigned int n = 128;

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {n}});
    auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {n}});
    mm->add_instruction(migraphx::make_op("add"), x, y);

    p.compile(migraphx::make_target("gpu"));

    std::vector<float> xdata(n, 4.0f);
    std::vector<float> ydata(n, 6.0f);
    auto xarg = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n}}, xdata.data()};
    auto yarg = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n}}, ydata.data()};

    auto gx = migraphx::gpu::to_gpu(xarg);
    auto gy = migraphx::gpu::to_gpu(yarg);

    migraphx::shape out_shape{migraphx::shape::float_type, {n}};
    auto out  = migraphx::fill_argument(out_shape, 0);
    auto gout = migraphx::gpu::to_gpu(out);

    migraphx::context& ctx_ref = p.get_context();
    auto* gpu_ctx              = ctx_ref.any_cast<migraphx::gpu::context>();
    EXPECT(gpu_ctx != nullptr);

    auto results = p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout}});

    EXPECT(not results.empty());
    EXPECT(not gpu_ctx->get_stream().has_external_stream());

    p.finish();
    auto host_result = migraphx::gpu::from_gpu(gout);
    verify_data(host_result, out_shape, 10.0f);
}

TEST_CASE(test_mixed_async_and_sync_evals)
{
    const unsigned int n = 128;

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {n}});
    auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {n}});
    mm->add_instruction(migraphx::make_op("add"), x, y);

    p.compile(migraphx::make_target("gpu"));

    std::vector<float> xdata(n, 1.0f);
    std::vector<float> ydata(n, 2.0f);
    auto xarg = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n}}, xdata.data()};
    auto yarg = migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n}}, ydata.data()};

    auto gx = migraphx::gpu::to_gpu(xarg);
    auto gy = migraphx::gpu::to_gpu(yarg);

    migraphx::shape out_shape{migraphx::shape::float_type, {n}};
    auto out  = migraphx::fill_argument(out_shape, 0);
    auto gout = migraphx::gpu::to_gpu(out);

    migraphx::context& ctx_ref = p.get_context();
    auto* gpu_ctx              = ctx_ref.any_cast<migraphx::gpu::context>();
    EXPECT(gpu_ctx != nullptr);

    auto ext = create_external_stream();

    // Async eval with external stream
    p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout}}, {ext.get(), true});
    EXPECT(not gpu_ctx->get_stream().has_external_stream());
    EXPECT(hipStreamSynchronize(ext.get()) == hipSuccess);

    auto host_result = migraphx::gpu::from_gpu(gout);
    verify_data(host_result, out_shape, 3.0f);

    // Sync eval with internal stream
    auto gout2 = migraphx::gpu::to_gpu(out);
    p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout2}});
    EXPECT(not gpu_ctx->get_stream().has_external_stream());
    p.finish();

    auto host_result2 = migraphx::gpu::from_gpu(gout2);
    verify_data(host_result2, out_shape, 3.0f);

    // Async eval again to confirm no stale state
    auto gout3 = migraphx::gpu::to_gpu(out);
    p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout3}}, {ext.get(), true});
    EXPECT(not gpu_ctx->get_stream().has_external_stream());
    EXPECT(hipStreamSynchronize(ext.get()) == hipSuccess);

    auto host_result3 = migraphx::gpu::from_gpu(gout3);
    verify_data(host_result3, out_shape, 3.0f);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
