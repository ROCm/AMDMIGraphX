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
    // A freshly-constructed context has no external binding.
    EXPECT(not stream.has_external_stream());

    auto ext = create_external_stream();
    stream.set_queue(ext.get());

    EXPECT(stream.get() == ext.get());
    EXPECT(stream.get() != internal);
    EXPECT(stream.has_external_stream());

    // Under std::optional semantics, set_queue(nullptr) does NOT
    // clear the binding: it rebinds to the HIP default stream (which is a
    // legal stream value).  has_external_stream() therefore stays true, and
    // get() now returns nullptr (= the default stream).
    stream.set_queue(nullptr);

    EXPECT(stream.get() == nullptr);
    EXPECT(stream.get() != internal);
    EXPECT(stream.has_external_stream());
}

TEST_CASE(test_stream_override_get_queue)
{
    migraphx::gpu::context ctx{};
    auto ext = create_external_stream();

    hipStream_t original_queue = ctx.get_queue().get<hipStream_t>();
    EXPECT(original_queue != nullptr);

    ctx.get_stream().set_queue(ext.get());
    EXPECT(ctx.get_queue().get<hipStream_t>() == ext.get());

    // Rebinding to nullptr means "the HIP default stream", not "no binding".
    // The active queue therefore changes value, but the external binding
    // remains in effect.  Use unsafe_get() to compare against nullptr:
    // context::get_queue() returns a default-constructed (untyped) any_ptr
    // when the bound stream is nullptr, so get<hipStream_t>() would throw a
    // type-mismatch exception on the empty name string.
    ctx.get_stream().set_queue(nullptr);

    EXPECT(ctx.get_queue().unsafe_get() == nullptr);
    EXPECT(ctx.get_queue().unsafe_get() != original_queue);
    EXPECT(ctx.get_stream().has_external_stream());
}

TEST_CASE(test_context_set_and_restore_queue)
{
    migraphx::gpu::context ctx{};
    auto ext = create_external_stream();

    migraphx::any_ptr queue(ext.get());

    hipStream_t before = ctx.get_queue().get<hipStream_t>();
    EXPECT(before != nullptr);

    // set_queue() (not wait_for) is what redirects the active binding.
    ctx.set_queue(queue);
    EXPECT(ctx.get_queue().get<hipStream_t>() == ext.get());
    EXPECT(ctx.get_queue().get<hipStream_t>() != before);
    EXPECT(ctx.get_stream().has_external_stream());

    // restore_queue() puts the original binding back; wait_for/finish_on
    // intentionally do not.
    ctx.restore_queue();
    EXPECT(ctx.get_queue().get<hipStream_t>() == before);
    EXPECT(not ctx.get_stream().has_external_stream());
}

TEST_CASE(test_context_wait_for_finish_on_do_not_rebind)
{
    migraphx::gpu::context ctx{};
    auto ext = create_external_stream();

    migraphx::any_ptr queue(ext.get());

    hipStream_t before = ctx.get_queue().get<hipStream_t>();

    // wait_for() is pure event sync; it must NOT mutate the active binding.
    ctx.wait_for(queue);
    EXPECT(ctx.get_queue().get<hipStream_t>() == before);
    EXPECT(not ctx.get_stream().has_external_stream());

    ctx.finish_on(queue);
    EXPECT(ctx.get_queue().get<hipStream_t>() == before);
    EXPECT(not ctx.get_stream().has_external_stream());
}

TEST_CASE(test_context_restore_queue_is_noop_when_unsaved)
{
    migraphx::gpu::context ctx{};
    hipStream_t before = ctx.get_queue().get<hipStream_t>();

    // Safe to call without a prior set_queue() -- the async epilogue in
    // program::eval relies on this.
    ctx.restore_queue();
    EXPECT(ctx.get_queue().get<hipStream_t>() == before);
    EXPECT(not ctx.get_stream().has_external_stream());
}

TEST_CASE(test_context_set_queue_with_null_then_restore)
{
    migraphx::gpu::context ctx{};
    auto ext = create_external_stream();

    // Pre-bind an external stream so the "original" binding under test is
    // not the internal stream.
    ctx.get_stream().set_queue(ext.get());
    EXPECT(ctx.get_queue().get<hipStream_t>() == ext.get());

    // nullptr is a *valid* queue value -- it binds the HIP default stream.
    // Under std::optional<hipStream_t> semantics the binding is still
    // active (has_external_stream() == true); the value is just nullptr.
    // set_queue(null) must NOT be conflated with restore.  Read back via
    // unsafe_get() because context::get_queue() returns an untyped any_ptr
    // when the bound stream is nullptr.
    ctx.set_queue(migraphx::any_ptr{});
    EXPECT(ctx.get_queue().unsafe_get() == nullptr);
    EXPECT(ctx.get_queue().unsafe_get() != ext.get());
    EXPECT(ctx.get_stream().has_external_stream());

    // restore_queue() unconditionally unbinds the external stream and routes
    // submissions back to the internal stream -- it does NOT replay the
    // previously-bound `ext` value.  Callers that need to re-establish a
    // prior external binding must call set_queue() themselves.
    ctx.restore_queue();
    EXPECT(not ctx.get_stream().has_external_stream());
    EXPECT(ctx.get_queue().get<hipStream_t>() != ext.get());
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
    // When the context had NO external binding prior to async eval, the
    // epilogue's restore_queue() must put the context back into the
    // "no binding" state -- the caller's transient stream must not leak.
    // This requires previous_stream to distinguish "no save" from
    // "saved (nothing bound)".
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
    EXPECT(not gpu_ctx->get_stream().has_external_stream());

    hipStream_t internal_stream = gpu_ctx->get_queue().get<hipStream_t>();

    p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout}}, {ext.get(), true});

    EXPECT(hipStreamSynchronize(ext.get()) == hipSuccess);

    EXPECT(not gpu_ctx->get_stream().has_external_stream());
    EXPECT(gpu_ctx->get_queue().get<hipStream_t>() == internal_stream);
}

TEST_CASE(test_external_stream_eval_unbinds_prior_binding)
{
    // program::eval's async epilogue calls restore_queue(), which under the
    // current contract unconditionally unbinds whatever external stream was
    // active at the start of eval -- including a binding the caller had
    // installed *before* eval.  Callers that want to keep a prior binding
    // alive across async evals must re-install it themselves; eval is not
    // responsible for preserving it.
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

    auto prior = create_external_stream();
    auto ext   = create_external_stream();

    migraphx::context& ctx_ref = p.get_context();
    auto* gpu_ctx              = ctx_ref.any_cast<migraphx::gpu::context>();
    EXPECT(gpu_ctx != nullptr);

    gpu_ctx->get_stream().set_queue(prior.get());
    EXPECT(gpu_ctx->get_queue().get<hipStream_t>() == prior.get());

    p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout}}, {ext.get(), true});

    EXPECT(hipStreamSynchronize(ext.get()) == hipSuccess);

    // Async eval's restore_queue() unconditionally drops the external
    // binding.  Neither the caller's `ext` nor the previously-bound
    // `prior` remains in effect.
    EXPECT(not gpu_ctx->get_stream().has_external_stream());
    EXPECT(gpu_ctx->get_queue().get<hipStream_t>() != ext.get());
    EXPECT(gpu_ctx->get_queue().get<hipStream_t>() != prior.get());
}

TEST_CASE(test_wait_for_finish_on_require_typed_queue)
{
    // The earlier "null-stream event fallback" was intentionally removed:
    // wait_for()/finish_on() are now pure event-sync primitives that assume
    // the caller has provided a typed (hipStream_t) any_ptr.  Passing a
    // default-constructed any_ptr is a programmer error and surfaces as a
    // type-mismatch exception, rather than silently no-op'ing.
    //
    // The async eval path in program::eval no longer calls wait_for() /
    // finish_on() at all -- it relies on set_queue()/restore_queue() for
    // queue rebinding -- so this is purely a direct-API contract test.
    migraphx::gpu::context ctx{};

    hipStream_t internal_before = ctx.get_queue().get<hipStream_t>();

    bool threw_on_wait_for = false;
    try
    {
        ctx.wait_for(migraphx::any_ptr{});
    }
    catch(const migraphx::exception&)
    {
        threw_on_wait_for = true;
    }
    EXPECT(threw_on_wait_for);

    bool threw_on_finish_on = false;
    try
    {
        ctx.finish_on(migraphx::any_ptr{});
    }
    catch(const migraphx::exception&)
    {
        threw_on_finish_on = true;
    }
    EXPECT(threw_on_finish_on);

    // The active binding is untouched on the error path.
    EXPECT(not ctx.get_stream().has_external_stream());
    EXPECT(ctx.get_queue().get<hipStream_t>() == internal_before);
}

TEST_CASE(test_async_eval_with_null_queue_uses_default_stream)
{
    // A default-constructed any_ptr is treated as "bind the HIP default
    // stream (nullptr)" by context::set_queue() -- not as a request for an
    // event fallback (there isn't one anymore).  The eval must dispatch on
    // the default stream and produce correct results once that stream is
    // synchronized.
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
    // Interleave async (caller-supplied stream) and sync (default) evals on
    // the same program and verify each produces correct results.  Async
    // eval's restore_queue() unconditionally unbinds the external stream,
    // so any caller-installed prior binding must be re-installed between
    // async cycles.  The sync block exercises the (uncommon but legal)
    // sync-eval-with-pre-bound-external-stream path, which depends on
    // wait() syncing the bound external stream rather than the internal one.
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

    auto prior = create_external_stream();
    auto ext   = create_external_stream();

    gpu_ctx->get_stream().set_queue(prior.get());
    EXPECT(gpu_ctx->get_queue().get<hipStream_t>() == prior.get());

    // Async eval with caller-supplied stream.  The async epilogue unbinds
    // whatever was bound -- including `prior` -- after eval.
    p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout}}, {ext.get(), true});
    EXPECT(hipStreamSynchronize(ext.get()) == hipSuccess);
    EXPECT(not gpu_ctx->get_stream().has_external_stream());

    auto host_result = migraphx::gpu::from_gpu(gout);
    verify_data(host_result, out_shape, 3.0f);

    // Re-bind `prior` so the sync eval below runs on it.  Sync eval does
    // not touch the queue binding, so kernels submit to `prior` and
    // p.finish() syncs `prior` (via wait()'s external_stream preference).
    gpu_ctx->get_stream().set_queue(prior.get());
    auto gout2 = migraphx::gpu::to_gpu(out);
    p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout2}});
    EXPECT(gpu_ctx->get_queue().get<hipStream_t>() == prior.get());
    p.finish();

    auto host_result2 = migraphx::gpu::from_gpu(gout2);
    verify_data(host_result2, out_shape, 3.0f);

    // Another async eval; again the caller's stream is unbound on exit
    // and so is `prior` (which was still bound coming in).
    auto gout3 = migraphx::gpu::to_gpu(out);
    p.eval({{"x", gx}, {"y", gy}, {"main:#output_0", gout3}}, {ext.get(), true});
    EXPECT(hipStreamSynchronize(ext.get()) == hipSuccess);
    EXPECT(not gpu_ctx->get_stream().has_external_stream());

    auto host_result3 = migraphx::gpu::from_gpu(gout3);
    verify_data(host_result3, out_shape, 3.0f);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
